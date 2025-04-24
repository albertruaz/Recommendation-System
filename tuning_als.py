"""
ALS 기반 추천 시스템 하이퍼파라미터 튜닝 스크립트
"""

import os
import json
import pandas as pd
import numpy as np
import itertools
import time
import gc
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

# PySpark 설정을 위한 환경 변수 설정
def configure_spark_env():
    """Spark 실행을 위한 환경 변수 설정"""
    # 메모리 설정
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-memory 4g --executor-memory 4g pyspark-shell'
    os.environ['SPARK_LOCAL_DIRS'] = '/tmp'
    
    # GC 로그 출력 완전 비활성화 (콘솔 출력만 제거)
    os.environ['SPARK_JAVA_OPTS'] = '-XX:+UseG1GC -XX:+UseCompressedOops -XX:-PrintGCDetails -XX:-PrintGCTimeStamps'
    
    # 로깅 레벨 설정 (콘솔 출력에 대해서만 ERROR로 설정)
    os.environ['SPARK_LOG_LEVEL'] = 'ERROR'
    
    # log4j.properties 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log4j_path = os.path.join(current_dir, "log4j.properties")
    
    # 해당 경로에 파일이 실제로 존재하는지 확인
    if os.path.exists(log4j_path):
        # log4j 설정 파일 경로 설정
        system_properties = [
            f"-Dlog4j.configuration=file:{log4j_path}",
            "-Dlog4j.debug=false"  # log4j 디버그 모드 비활성화
        ]
        
        if 'SPARK_JAVA_OPTS' in os.environ:
            os.environ['SPARK_JAVA_OPTS'] += ' ' + ' '.join(system_properties)
        else:
            os.environ['SPARK_JAVA_OPTS'] = ' '.join(system_properties)
    
    # SparkUI 비활성화 - 콘솔 출력 제거를 위한 설정
    spark_conf = [
        "--conf spark.ui.enabled=false",
        "--conf spark.ui.showConsoleProgress=false"
    ]
    
    if 'PYSPARK_SUBMIT_ARGS' in os.environ:
        submit_args = os.environ['PYSPARK_SUBMIT_ARGS']
        # pyspark-shell 앞에 설정 추가
        if "pyspark-shell" in submit_args:
            parts = submit_args.split("pyspark-shell")
            os.environ['PYSPARK_SUBMIT_ARGS'] = f"{parts[0]} {' '.join(spark_conf)} pyspark-shell{parts[1] if len(parts) > 1 else ''}"
        else:
            os.environ['PYSPARK_SUBMIT_ARGS'] += ' ' + ' '.join(spark_conf)

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

def run_model(model_type, params, interactions_df, top_n, run=None):
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
            # PySpark ALS 모델을 사용하기 전에 Spark 설정
            configure_spark_env()
            
            # import SparkSession을 함수 내부에서 수행
            from pyspark.sql import SparkSession
            # 기존 Spark 세션이 있으면 종료
            spark = SparkSession._instantiatedSession
            if spark is not None:
                spark.stop()
                # 명시적으로 GC 실행
                gc.collect()
                time.sleep(2)  # 리소스가 해제될 시간을 줌
            
            # PySpark ALS 모델 임포트
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
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            TUNING_CONFIG['default_params']['output_dir'],
            f"{model_type}_iter{params['max_iter']}_reg{params['reg_param']}_rank{params['rank']}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # 성능 지표 수집
        metrics = {
            'prep_time': prep_time,
            'train_time': train_time,
            'total_time': prep_time + train_time,
            'output_dir': output_dir
        }
        
        # 모델 구현별로 다른 성능 지표 수집
        if hasattr(model, 'metrics'):
            metrics.update(model.metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"모델 실행 오류: {str(e)}")
        raise
    finally:
        # 리소스 정리
        if 'model' in locals():
            try:
                model.cleanup()
            except Exception as cleanup_err:
                logger.warning(f"모델 리소스 정리 중 오류: {str(cleanup_err)}")
                
            # 모델 객체 명시적으로 제거
            del model
        
        # 명시적 GC 호출
        gc.collect()
        time.sleep(1)  # 메모리 해제될 시간 제공

def run_model_optimized(model_type, params, interactions_df, matrix_data):
    """행렬 변환 결과를 재사용하여 최적화된 모델 학습 및 평가 실행"""
    try:
        logger.info(f"모델 파라미터: {params}")
        
        # 모델 초기화
        if model_type == 'pyspark_als':
            # PySpark ALS 모델을 사용하기 전에 Spark 설정
            configure_spark_env()
            
            # import SparkSession을 함수 내부에서 수행
            from pyspark.sql import SparkSession
            # 기존 Spark 세션이 있으면 종료
            spark = SparkSession._instantiatedSession
            if spark is not None:
                spark.stop()
                # 명시적으로 GC 실행
                gc.collect()
                time.sleep(2)  # 리소스가 해제될 시간을 줌
            
            # PySpark ALS 모델 임포트
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
        else:
            logger.error("이 함수는 pyspark_als 모델 유형에만 사용할 수 있습니다.")
            raise ValueError("Invalid model_type for optimized run")
        
        # 행렬 데이터가 이미 준비되어 있으므로 prepare_matrices 단계 생략
        # 대신 ALS 클래스의 인덱스 매핑 정보 설정 (prepare_matrices에서 수행하는 일부 작업)
        model._prepare_indices_from_matrix_data(matrix_data)
        
        # 모델 학습 (이미 준비된 행렬 데이터 사용)
        start_time = time.time()
        model.train(interactions_df=interactions_df, matrix_data=matrix_data)
        train_time = time.time() - start_time
        logger.info(f"모델 학습 완료 (소요 시간: {train_time:.2f}초)")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            TUNING_CONFIG['default_params']['output_dir'],
            f"{model_type}_iter{params['max_iter']}_reg{params['reg_param']}_rank{params['rank']}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # 성능 지표 수집
        metrics = {
            'train_time': train_time,
            'total_time': train_time,  # 행렬 변환 시간 제외됨
            'output_dir': output_dir
        }
        
        # 모델 구현별로 다른 성능 지표 수집
        if hasattr(model, 'metrics'):
            metrics.update(model.metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"모델 실행 오류: {str(e)}")
        raise
    finally:
        # 리소스 정리
        if 'model' in locals():
            try:
                model.cleanup()
            except Exception as cleanup_err:
                logger.warning(f"모델 리소스 정리 중 오류: {str(cleanup_err)}")
                
            # 모델 객체 명시적으로 제거
            del model
        
        # 명시적 GC 호출
        gc.collect()
        time.sleep(1)  # 메모리 해제될 시간 제공

def main():
    # 설정 로드
    logger.info("하이퍼파라미터 튜닝 시작")
    days = TUNING_CONFIG['default_params']['days']
    top_n = TUNING_CONFIG['default_params']['top_n']
    
    try:
        # Wandb 초기화 - API 키는 환경 변수에서 가져옴
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            logger.warning("WANDB_API_KEY 환경 변수가 설정되지 않았습니다. wandb 로그인이 필요할 수 있습니다.")
            wandb.login(relogin=False)  # 필요한 경우에만 로그인 프롬프트 표시
        else:
            # API 키가 있으면 자동 로그인
            wandb.login(key=wandb_api_key, relogin=True)
            logger.info("wandb에 자동 로그인되었습니다.")
        
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
        
        # Wandb 프로젝트 초기화 - 새 이름 사용
        wandb_project = f"als-optimize-{model_type}-v2"
        
        # 모든 하이퍼파라미터 조합에 대해 모델 실행 (wandb 외부에서)
        all_params_list = []
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
                    
            all_params_list.append(params)
        
        # PySpark ALS 모델인 경우 메모리 문제 방지를 위해 하이퍼파라미터 규모 조정
        if model_type == 'pyspark_als':
            # 1. 배치 크기 설정 - 한 번에 처리할 최대 하이퍼파라미터 조합 수 제한
            batch_size = 3
            
            # 2. 메모리 소비가 적은 조합부터 실행하도록 정렬
            # 메모리 소비를 추정하는 함수 (rank가 클수록, max_iter가 클수록 메모리 소비 증가)
            def estimate_memory_usage(params):
                return params['rank'] * params['max_iter']
            
            all_params_list.sort(key=estimate_memory_usage)
            
            # 3. 행렬 변환 사전 계산 (공통 행렬 생성)
            # 단, interaction_weights가 모든 파라미터 세트에서 동일한 경우에만 가능
            if all(params['interaction_weights'] == all_params_list[0]['interaction_weights'] 
                   for params in all_params_list):
                logger.info("모든 파라미터 조합이 동일한 interaction_weights를 사용하므로 행렬 변환을 한 번만 수행합니다.")
                
                # 첫 번째 파라미터 세트로 모델 초기화 (matrix_data 생성용)
                from models.pyspark_als import PySparkALS
                matrix_model = PySparkALS(
                    max_iter=10,  # 임시 값, 행렬 변환에만 사용됨
                    reg_param=0.1, # 임시 값
                    rank=10,  # 임시 값
                    random_state=42,
                    interaction_weights=all_params_list[0]['interaction_weights']
                )
                
                # PySpark 설정
                configure_spark_env()
                
                # 행렬 변환 수행
                logger.info("공통 행렬 변환 시작...")
                start_time = time.time()
                common_matrix_data = matrix_model.prepare_matrices(interactions_df)
                prep_time = time.time() - start_time
                logger.info(f"공통 행렬 변환 완료 (소요 시간: {prep_time:.2f}초)")
                
                # 메모리 관리를 위해 모델 객체 제거
                matrix_model.cleanup()
                del matrix_model
                gc.collect()
                time.sleep(2)
                
                # 3. 배치 단위로 실행 (행렬 변환 재사용)
                for i in range(0, len(all_params_list), batch_size):
                    batch_params = all_params_list[i:i+batch_size]
                    for j, params in enumerate(batch_params):
                        logger.info(f"\n=== 하이퍼파라미터 튜닝 진행 상황 ===")
                        logger.info(f"전체 진행: [{i+j+1}/{len(all_params_list)}] ({((i+j+1)/len(all_params_list))*100:.1f}%)")
                        logger.info(f"현재 파라미터: max_iter={params['max_iter']}, reg_param={params['reg_param']}, rank={params['rank']}")
                        try:
                            # 최적화된 모델 실행 함수 호출 (matrix_data 재사용)
                            result = run_model_optimized(model_type, params, interactions_df, common_matrix_data)
                            result.update(params)
                            tuning_results.append(result)
                        except Exception as e:
                            logger.error(f"조합 {params} 실행 실패: {str(e)}")
                            # 실패한 경우에도 일부 정보 저장
                            result = {
                                'max_iter': params['max_iter'],
                                'reg_param': params['reg_param'],
                                'rank': params['rank'],
                                'error': str(e)
                            }
                            tuning_results.append(result)
                        
                        # 메모리 정리를 위한 시간 제공
                        gc.collect()
                        time.sleep(3)
            else:
                logger.info("interaction_weights가 파라미터 조합마다 다르므로 각각 행렬 변환을 수행합니다.")
                # 기존 방식대로 실행 (각 조합별로 행렬 변환 포함)
                for i in range(0, len(all_params_list), batch_size):
                    batch_params = all_params_list[i:i+batch_size]
                    for j, params in enumerate(batch_params):
                        logger.info(f"\n=== 하이퍼파라미터 튜닝 진행 상황 ===")
                        logger.info(f"전체 진행: [{i+j+1}/{len(all_params_list)}] ({((i+j+1)/len(all_params_list))*100:.1f}%)")
                        logger.info(f"현재 파라미터: max_iter={params['max_iter']}, reg_param={params['reg_param']}, rank={params['rank']}")
                        try:
                            result = run_model(model_type, params, interactions_df, top_n)
                            result.update(params)
                            tuning_results.append(result)
                        except Exception as e:
                            logger.error(f"조합 {params} 실행 실패: {str(e)}")
                            # 실패한 경우에도 일부 정보 저장
                            result = {
                                'max_iter': params['max_iter'],
                                'reg_param': params['reg_param'],
                                'rank': params['rank'],
                                'error': str(e)
                            }
                            tuning_results.append(result)
                        
                        # 메모리 정리를 위한 시간 제공
                        gc.collect()
                        time.sleep(3)
        else:
            # 다른 모델은 일반적인 방식으로 실행
            for i, params in enumerate(all_params_list):
                logger.info(f"파라미터 조합 실행 [{i+1}/{len(all_params_list)}]: {params}")
                result = run_model(model_type, params, interactions_df, top_n)
                result.update(params)
                tuning_results.append(result)
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(tuning_results)
        
        # 성능 지표 기준으로 결과 정렬
        if 'test_huber_loss' in results_df.columns:
            results_df = results_df.sort_values('test_huber_loss')
        elif 'test_rmse' in results_df.columns:
            results_df = results_df.sort_values('test_rmse')
        
        # 결과 파일 저장
        results_path = os.path.join(
            TUNING_CONFIG['default_params']['output_dir'],
            f"tuning_results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results_df.to_csv(results_path, index=False)
        logger.info(f"튜닝 결과가 {results_path}에 저장되었습니다.")
        
        # Wandb에 한 번에 결과 시각화 (단일 run으로)
        with wandb.init(project=wandb_project, entity="vingle", name=f"{model_type}-hyper-search", job_type="sweep") as run:
            # 설정한 모든 하이퍼파라미터 값을 config에 등록
            param_space = {key: model_config[key] for key in param_keys}
            run.config.update({
                "param_space": param_space,
                "num_trials": len(tuning_results),
                "model_type": model_type,
                "days": days,
                "top_n": top_n
            })
            
            # WandB Artifacts로 결과 CSV 저장
            results_artifact = wandb.Artifact(f"{model_type}-tuning-results", type="dataset")
            results_artifact.add_file(results_path)
            run.log_artifact(results_artifact)
            
            # 결과 테이블 생성 및 로깅
            table_columns = ["trial_id", "max_iter", "reg_param", "rank"]
            
            # 모델 성능 지표 열 추가
            if 'train_rmse' in results_df.columns:
                table_columns.extend(["train_rmse", "test_rmse"])
            if 'train_huber_loss' in results_df.columns:
                table_columns.extend(["train_huber_loss", "test_huber_loss"])
                
            table_columns.extend(["train_time", "total_time"])
            
            # Table 데이터 생성
            table_data = []
            for idx, row in results_df.iterrows():
                table_row = [idx]
                for col in table_columns[1:]:
                    if col in row:
                        table_row.append(row[col])
                    else:
                        table_row.append(None)
                table_data.append(table_row)
            
            results_table = wandb.Table(columns=table_columns, data=table_data)
            run.log({"hyperparameter_tuning_results": results_table})
            
            # Parallel Coordinates Plot
            # 주요 하이퍼파라미터와 성능 지표만 선택
            pc_columns = ["max_iter", "reg_param", "rank"]
            if 'test_rmse' in results_df.columns:
                pc_columns.append("test_rmse")
            if 'test_huber_loss' in results_df.columns:
                pc_columns.append("test_huber_loss")
            pc_columns.append("train_time")
            
            # 결과에서 필요한 열만 추출
            pc_data = []
            for _, row in results_df.iterrows():
                row_data = []
                for col in pc_columns:
                    if col in row:
                        row_data.append(row[col])
                    else:
                        row_data.append(None)
                pc_data.append(row_data)
            
            # Parallel Coordinates 차트 생성
            parallel_coords = wandb.plot.parallel_coordinates(
                table_data=pc_data,
                columns=pc_columns,
                title="Hyperparameter Tuning Comparison"
            )
            run.log({"parallel_coordinates": parallel_coords})
            
            # 하이퍼파라미터별 성능 비교 차트
            # 1. RMSE/Huber Loss를 기준으로 한 하이퍼파라미터 영향도 분석
            for param in ["max_iter", "reg_param", "rank"]:
                if param not in results_df.columns:
                    continue
                    
                param_values = sorted(results_df[param].unique())
                
                # 각 파라미터 값에 대한 성능 지표 평균 계산
                for metric in ["test_rmse", "test_huber_loss", "train_time"]:
                    if metric not in results_df.columns:
                        continue
                        
                    metric_by_param = []
                    for val in param_values:
                        subset = results_df[results_df[param] == val]
                        avg_metric = subset[metric].mean()
                        metric_by_param.append([str(val), avg_metric])
                    
                    # 차트 생성 및 로깅
                    param_table = wandb.Table(columns=[param, metric], data=metric_by_param)
                    run.log({f"{param}_vs_{metric}": wandb.plot.bar(param_table, param, metric,
                                                                    title=f"Impact of {param} on {metric}")})
            
            # 2. 최적의 하이퍼파라미터 조합 시각화
            if 'test_huber_loss' in results_df.columns and not results_df['test_huber_loss'].isna().all():
                best_idx = results_df['test_huber_loss'].fillna(float('inf')).argmin()
                best_metric = 'test_huber_loss'
            elif 'test_rmse' in results_df.columns and not results_df['test_rmse'].isna().all():
                best_idx = results_df['test_rmse'].fillna(float('inf')).argmin()
                best_metric = 'test_rmse'
            else:
                best_idx = results_df['train_time'].fillna(float('inf')).argmin()
                best_metric = 'train_time'
                
            best_config = results_df.iloc[best_idx]
            best_params = {param: best_config[param] for param in param_keys if param in best_config}
            
            # 최적 파라미터 로깅
            run.summary["best_parameters"] = best_params
            run.summary["best_" + best_metric] = best_config.get(best_metric)
            
            # 다양한 지표를 기준으로 최적 파라미터 추출
            if 'test_huber_loss' in results_df.columns and not results_df['test_huber_loss'].isna().all():
                best_huber = results_df.iloc[results_df['test_huber_loss'].fillna(float('inf')).argmin()]
                logger.info(f"Huber Loss 기준 최적 파라미터: {best_huber[param_keys].to_dict()}")
                logger.info(f"Best Huber Loss: {best_huber['test_huber_loss']:.4f}")
                run.summary["best_huber_params"] = best_huber[param_keys].to_dict()
                run.summary["best_huber_loss"] = best_huber['test_huber_loss']
            
            if 'test_rmse' in results_df.columns and not results_df['test_rmse'].isna().all():
                best_rmse = results_df.iloc[results_df['test_rmse'].fillna(float('inf')).argmin()]
                logger.info(f"RMSE 기준 최적 파라미터: {best_rmse[param_keys].to_dict()}")
                logger.info(f"Best RMSE: {best_rmse['test_rmse']:.4f}")
                run.summary["best_rmse_params"] = best_rmse[param_keys].to_dict()
                run.summary["best_rmse"] = best_rmse['test_rmse']
            
            if 'train_time' in results_df.columns and not results_df['train_time'].isna().all():
                best_time = results_df.iloc[results_df['train_time'].fillna(float('inf')).argmin()]
                logger.info(f"학습 시간 기준 최적 파라미터: {best_time[param_keys].to_dict()}")
                logger.info(f"Best Train Time: {best_time['train_time']:.2f}초")
                run.summary["best_time_params"] = best_time[param_keys].to_dict()
                run.summary["best_train_time"] = best_time['train_time']
            
            # 3D 시각화 (3개의 주요 성능 지표에 대해)
            valid_metrics = all(
                metric in results_df.columns and not results_df[metric].isna().all()
                for metric in ['test_rmse', 'test_huber_loss', 'train_time']
            )
            
            if valid_metrics:
                # NaN 값을 제외하고 처리
                scatter_df = results_df.dropna(subset=['test_rmse', 'test_huber_loss', 'train_time'])
                if not scatter_df.empty:
                    scatter_data = [[row['test_rmse'], row['test_huber_loss'], row['train_time']] 
                                   for _, row in scatter_df.iterrows()]
                    scatter_table = wandb.Table(data=scatter_data, columns=["test_rmse", "test_huber_loss", "train_time"])
                    run.log({"performance_3d_scatter": wandb.plot_table(
                        "wandb/scatter/v0", scatter_table, {"x": "test_rmse", "y": "test_huber_loss", "z": "train_time"},
                        {"title": "Performance Trade-offs (RMSE vs Huber Loss vs Train Time)"}
                    )})
            
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 