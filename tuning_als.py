"""
ALS 기반 추천 시스템 하이퍼파라미터 튜닝 스크립트
"""

import os
import json
import pandas as pd
import numpy as np
import itertools
import time
from datetime import datetime
import wandb
from dotenv import load_dotenv
from database.recommendation_db import RecommendationDB
from database.excel_db import ExcelRecommendationDB

from utils.logger import setup_logger

# 환경 변수 로드
load_dotenv()

# 튜닝 설정 파일 로드
with open('config/tuning_config.json', 'r') as f:
    TUNING_CONFIG = json.load(f)

# 로거 설정
logger = setup_logger('tuning')

def load_interactions(days: int = 30, use_test_db: bool = False) -> pd.DataFrame:
    """데이터베이스에서 상호작용 데이터 로드"""
    logger.info(f"days: {days}")
    
    if use_test_db:
        db = ExcelRecommendationDB()
    else:
        db = RecommendationDB()
        
    interactions = db.get_user_item_interactions(days=days)
    
    if interactions.empty:
        logger.error(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
        raise ValueError(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
    
    logger.info(f"총 {len(interactions)}개의 상호작용 데이터 로드 완료")
    return interactions

def run_model(model_type, params, interactions_df, top_n, run):
    """모델 학습 및 평가 실행"""
    try:
        logger.info(f"모델 파라미터: {params}")
        
        # 모델 초기화
        if model_type == 'custom_implicit_als':
            from models.custom_implicit_als import CustomImplicitALS
            model = CustomImplicitALS(
                max_iter=params['max_iter'],
                reg_param=params['reg_param'],
                rank=params['rank'],
                random_state=42,
                interaction_weights=params['interaction_weights']
            )
        elif model_type == 'implicit_als':
            from models.implicit_als import ImplicitALS
            model = ImplicitALS(
                max_iter=params['max_iter'],
                reg_param=params['reg_param'],
                rank=params['rank'],
                random_state=42,
                alpha=params.get('alpha', 40),
                interaction_weights=params['interaction_weights']
            )
        elif model_type == 'pyspark_als':
            from models.pyspark_als import PySparkALS
            model = PySparkALS(
                max_iter=params['max_iter'],
                reg_param=params['reg_param'],
                rank=params['rank'],
                random_state=42,
                alpha=params.get('alpha', 1.0),
                interaction_weights=params['interaction_weights'],
                max_prediction=50.0,  # 예측값 상한 설정
                huber_delta=10.0      # Huber Loss의 델타값 설정
            )
        else:  # buffalo_als
            from models.buffalo_als import BuffaloALS
            model = BuffaloALS(
                max_iter=params['max_iter'],
                reg_param=params['reg_param'],
                rank=params['rank'],
                random_state=42,
                alpha=params.get('alpha', 40),
                interaction_weights=params['interaction_weights']
            )
        
        # 행렬 데이터 준비
        start_time = time.time()
        matrix_data = model.prepare_matrices(interactions_df)
        prep_time = time.time() - start_time
        logger.info(f"행렬 변환 완료 (소요 시간: {prep_time:.2f}초)")
        
        # 모델 학습
        start_time = time.time()
        model.train(interactions_df=interactions_df, matrix_data=matrix_data)
        train_time = time.time() - start_time
        logger.info(f"모델 학습 완료 (소요 시간: {train_time:.2f}초)")
        
        # 추천 생성
        start_time = time.time()
        recommendations_df = model.generate_recommendations(top_n=top_n)
        rec_time = time.time() - start_time
        logger.info(f"추천 생성 완료 (소요 시간: {rec_time:.2f}초)")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            TUNING_CONFIG['default_params']['output_dir'],
            f"{model_type}_iter{params['max_iter']}_reg{params['reg_param']}_rank{params['rank']}"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'recommendations_{timestamp}.csv')
        recommendations_df.to_csv(output_path, index=False)
        logger.info(f"추천 결과가 {output_path}에 저장되었습니다.")
        
        # 성능 지표 수집
        metrics = {
            'prep_time': prep_time,
            'train_time': train_time,
            'rec_time': rec_time,
            'total_time': prep_time + train_time + rec_time,
            'output_path': output_path
        }
        
        # 모델 구현별로 다른 성능 지표 수집
        if hasattr(model, 'metrics'):
            metrics.update(model.metrics)
        
        # Wandb에 기록
        if run is not None:
            log_metrics = {
                'train_time': train_time,
                'total_time': metrics['total_time']
            }
            
            # 모델별 성능 지표 기록
            if 'train_rmse' in metrics:
                log_metrics['train_rmse'] = metrics['train_rmse']
            if 'test_rmse' in metrics:
                log_metrics['test_rmse'] = metrics['test_rmse']
            if 'train_huber_loss' in metrics:
                log_metrics['train_huber_loss'] = metrics['train_huber_loss']
            if 'test_huber_loss' in metrics:
                log_metrics['test_huber_loss'] = metrics['test_huber_loss']
            
            # 추천 상위 10개의 유사도 평균 계산 (선택적)
            if len(recommendations_df) > 0:
                sample = recommendations_df.head(1000)
                avg_pred_rating = sample['predicted_rating'].mean()
                log_metrics['avg_pred_rating'] = avg_pred_rating
            
            run.log(log_metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"모델 실행 오류: {str(e)}")
        raise
    finally:
        if 'model' in locals():
            model.cleanup()

def main():
    # 설정 로드
    logger.info("하이퍼파라미터 튜닝 시작")
    days = TUNING_CONFIG['default_params']['days']
    top_n = TUNING_CONFIG['default_params']['top_n']
    
    try:
        # Wandb 초기화
        wandb.login(relogin=True)  # vingle 팀에 접근할 수 있는 API 키로 로그인
        
        # 데이터 로드
        interactions_df = load_interactions(days=days, use_test_db=True)
        
        # 모델 타입 확인
        model_type = TUNING_CONFIG['model_type']
        logger.info(f"튜닝할 모델: {model_type}")
        
        # 모델 설정 가져오기
        model_config = TUNING_CONFIG[model_type]
        
        # 하이퍼파라미터 조합 생성
        param_keys = []
        param_values = []
        
        for key, value in model_config.items():
            if isinstance(value, list):
                param_keys.append(key)
                param_values.append(value)
        
        # 튜닝 결과를 저장할 리스트
        tuning_results = []
        
        # Wandb 프로젝트 초기화
        wandb_project = f"als-tuning-{model_type}"
        
        # 모든 하이퍼파라미터 조합에 대해 모델 실행
        for params_tuple in itertools.product(*param_values):
            # 파라미터 딕셔너리 생성
            params = {
                param_keys[i]: params_tuple[i] 
                for i in range(len(param_keys))
            }
            
            # 고정 파라미터 추가
            for key, value in model_config.items():
                if key not in params:
                    params[key] = value
            
            # wandb run 시작
            run_name = f"{model_type}-iter{params['max_iter']}-reg{params['reg_param']}-rank{params['rank']}"
            
            with wandb.init(project=wandb_project, entity="vingle", name=run_name, config=params, reinit=True) as run:
                # 모델 실행
                logger.info(f"파라미터 조합 실행: {params}")
                result = run_model(model_type, params, interactions_df, top_n, run)
                
                # 결과 저장
                result.update(params)
                tuning_results.append(result)
        
        # 튜닝 결과 저장
        results_df = pd.DataFrame(tuning_results)
        results_path = os.path.join(
            TUNING_CONFIG['default_params']['output_dir'],
            f"tuning_results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results_df.to_csv(results_path, index=False)
        logger.info(f"튜닝 결과가 {results_path}에 저장되었습니다.")
        
        # 다양한 지표를 기준으로 최적 파라미터 출력
        if 'test_huber_loss' in results_df.columns:
            best_huber = results_df.iloc[results_df['test_huber_loss'].argmin()]
            logger.info(f"Huber Loss 기준 최적 파라미터: {best_huber[param_keys].to_dict()}")
            logger.info(f"Best Huber Loss: {best_huber['test_huber_loss']:.4f}")
        
        if 'test_rmse' in results_df.columns:
            best_rmse = results_df.iloc[results_df['test_rmse'].argmin()]
            logger.info(f"RMSE 기준 최적 파라미터: {best_rmse[param_keys].to_dict()}")
            logger.info(f"Best RMSE: {best_rmse['test_rmse']:.4f}")
        
        best_time = results_df.iloc[results_df['train_time'].argmin()]
        logger.info(f"학습 시간 기준 최적 파라미터: {best_time[param_keys].to_dict()}")
        logger.info(f"Best Train Time: {best_time['train_time']:.2f}초")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 