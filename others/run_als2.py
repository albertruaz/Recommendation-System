"""
ALS 기반 추천 시스템 실행 클래스
"""

import os
import json
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from database.db import db
from model_als.pyspark_als import PySparkALS
from utils.logger import setup_logger, overall_log
from utils.recommendation_utils import save_recommendations

from sklearn.model_selection import train_test_split

class RunALSTuning:
    def __init__(self, max_iter, reg_param, rank):
        # 설정 파일 로드
        with open('config/als_config.json', 'r') as f:
            self.als_config = json.load(f)
        self.als_config['pyspark_als']['max_iter'] = max_iter
        self.als_config['pyspark_als']['reg_param'] = reg_param
        self.als_config['pyspark_als']['rank'] = rank
        
        # 로거 설정
        self.logger = setup_logger('run_als')
        
        # 기본 설정 로드
        self.days = self.als_config['default_params']['days']
        self.top_n = self.als_config['default_params']['top_n']
        self.output_dir = self.als_config['default_params']['output_dir']
        
        # 테스트 설정 로드
        self.split_test_data = self.als_config['testing'].get('split_test_data', False)
        self.test_ratio = self.als_config['testing'].get('test_ratio', 0.2)
        self.random_seed = self.als_config['testing'].get('random_seed', 42)
        self.enable_loss_calculation = self.als_config['testing'].get('calculate_loss', True)
        
        self.model = None
        self.test_df = None
        self.train_df = None

    def load_interactions(self) -> pd.DataFrame:
        """상호작용 데이터 로드"""
        db_instance = db()  
        interactions = db_instance.get_user_item_interactions(days=self.days, use_cache=True)
        
        if interactions.empty:
            self.logger.error(f"최근 {self.days}일 간의 상호작용 데이터가 없습니다.")
            raise ValueError(f"최근 {self.days}일 간의 상호작용 데이터가 없습니다.")
        self.logger.info(f"총 {len(interactions)}개의 상호작용 데이터 로드 완료")

        return interactions
    
    def split_data(self, interactions_df):
        """데이터 학습/테스트 분할"""
        if not self.split_test_data:
            return interactions_df, None
        
        train_df, test_df = train_test_split(
            interactions_df, 
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=interactions_df['interaction_type'] if 'interaction_type' in interactions_df.columns else None
        )
        
        return train_df, test_df
    
    def calculate_loss(self, data_df, data_type='test'):
        """데이터에 대한 손실 계산 및 예측 결과 생성"""
        if data_df is None or data_df.empty or self.model is None:
            return None
        try:

            valid_predictions = data_df.dropna(subset=['prediction'])
            
            if len(valid_predictions) == 0:
                return {
                    "mae": float('nan'), 
                    "rmse": float('nan'),
                    "same_direction": float('nan'),
                    "samples": 0
                }
            
            actual = valid_predictions['rating'].values
            pred = valid_predictions['prediction'].values
            
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean(np.square(actual - pred)))
            # 방향성 지표 계산 - 2를 기준으로 actual과 prediction이 같은 방향에 있는지 확인
            # (actual > 2 and pred > 2) OR (actual < 2 and pred < 2) OR (actual == 2 and pred == 2)
            same_direction = np.mean(
                ((actual > 2) & (pred > 2)) | 
                ((actual < 2) & (pred < 2)) | 
                ((actual == 2) & (pred == 2))
            )
            
            result_path = None
            if data_type == 'test':
                os.makedirs(self.output_dir, exist_ok=True)
                result_path = os.path.join(self.output_dir, f'{data_type}_predictions_{self.days}days.csv')
                data_df.to_csv(result_path, index=False)
                self.logger.info(f"{data_type.capitalize()} 예측 결과가 {result_path}에 저장되었습니다.")
            
            result = {
                "mae": float(mae),
                "rmse": float(rmse),
                "same_direction": float(same_direction),
                "samples": len(valid_predictions)
            }
            
            if result_path:
                result["results_path"] = result_path
                
            return result
            
        except Exception as e:
            self.logger.error(f"{data_type.capitalize()} 손실 계산 중 오류 발생: {str(e)}")
            return None
    
    def run(self):
        """ALS 모델 실행 및 추천 생성"""
        self.logger.info(f"Start ALS")
        
        try:
            # 실행 ID 생성 (현재 날짜/시간 + UUID)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]  # UUID의 첫 8자만 사용
            run_id = f"{timestamp}_{unique_id}"
            output_dir_with_id = os.path.join(self.output_dir, run_id)
            os.makedirs(output_dir_with_id, exist_ok=True)
            
            
            interactions_df = self.load_interactions()
            self.train_df, self.test_df = self.split_data(interactions_df)
            
            self.model = PySparkALS(
                max_iter=self.als_config['pyspark_als']['max_iter'],
                reg_param=self.als_config['pyspark_als']['reg_param'],
                rank=self.als_config['pyspark_als']['rank'],
                interaction_weights=self.als_config['pyspark_als']['interaction_weights'],
                max_prediction=50.0,
                huber_delta=10.0,
                split_test_data=self.split_test_data
            )
            
            # 전체 데이터로 인덱스 구성, 학습 데이터만 사용하여 학습
            matrix_data = self.model.prepare_matrices(interactions_df, train_df=self.train_df)
            
            # 모델 학습
            self.logger.info("모델 학습 시작")
            self.model.train(matrix_data=matrix_data)
            
            # 추천 생성
            recommendations_df = self.model.generate_recommendations(top_n=self.top_n)
            
            train_result = None
            test_result = None
            if self.enable_loss_calculation:
                self.train_predictions = self.model.test(self.train_df)
                train_result = self.calculate_loss(self.train_predictions, 'train')
            if self.split_test_data and self.enable_loss_calculation and self.test_df is not None:
                self.test_predictions = self.model.test(self.test_df)
                test_result = self.calculate_loss(self.test_predictions, 'test')
            
            
            # 결과 반환
            result = {
                "recommendations": recommendations_df,
                "train_result": train_result,
                "test_result": test_result
            }

            if train_result is not None:
                self.logger.info(f"학습 데이터 결과:")
                self.logger.info(f"- MAE: {train_result['mae']:.4f}")
                self.logger.info(f"- RMSE: {train_result['rmse']:.4f}")
                self.logger.info(f"- 방향성 일치도: {train_result['same_direction']:.4f}")
                self.logger.info(f"- 샘플 수: {train_result['samples']}")
            
            # 테스트 결과 정보 출력 (있는 경우)
            if test_result is not None:
                self.logger.info(f"테스트 데이터 결과:")
                self.logger.info(f"- MAE: {test_result['mae']:.4f}")
                self.logger.info(f"- RMSE: {test_result['rmse']:.4f}")
                self.logger.info(f"- 방향성 일치도: {test_result['same_direction']:.4f}")
                self.logger.info(f"- 샘플 수: {test_result['samples']}")
            save_recommendations(recommendations_df, output_dir=output_dir_with_id)
            # 전체 실행 로그 저장 - run_id 전달
            overall_log(run_id, train_result, test_result, self.als_config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"오류 발생: {str(e)}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if self.model is not None:
            self.model.cleanup()
            self.model = None
        self.test_data = None

# 직접 실행할 경우의 예시
if __name__ == "__main__":
    # 기본 하이퍼파라미터 설정
    max_iter = 50
    reg_param = 0.01
    rank = 10
    
    als_runner = RunALSTuning(max_iter, reg_param, rank)
    result = als_runner.run()
    if 'recommendations' in result:
        print(f"추천 결과: {len(result['recommendations'])}개")
    if 'train_result' in result and result['train_result'] is not None:
        print(f"훈련 데이터 MAE: {result['train_result']['mae']:.4f}")
        print(f"훈련 데이터 RMSE: {result['train_result']['rmse']:.4f}")
        print(f"훈련 데이터 방향성 일치도: {result['train_result']['same_direction']:.4f}")
    if 'test_result' in result and result['test_result'] is not None:
        print(f"테스트 데이터 MAE: {result['test_result']['mae']:.4f}")
        print(f"테스트 데이터 RMSE: {result['test_result']['rmse']:.4f}")
        print(f"테스트 데이터 방향성 일치도: {result['test_result']['same_direction']:.4f}") 