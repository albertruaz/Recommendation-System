"""
PySpark ALS 기반 추천 시스템 하이퍼파라미터 튜닝 스크립트
"""

import os
import json
import pandas as pd
import numpy as np
import itertools
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from database.excel_db import ExcelRecommendationDB

from utils.logger import setup_logger
from utils.spark_utils import SparkSingleton, safe_format, calculate_huber_loss, evaluate_model, apply_prediction_cap, calculate_validation_weighted_sum_spark

# 설정 파일 로드
with open('config/tuning_config.json', 'r') as f:
    TUNING_CONFIG = json.load(f)

# 로거 설정
logger = setup_logger('tuning')

def load_interactions(days: int = 30, use_test_db: bool = True) -> pd.DataFrame:
    """데이터베이스에서 상호작용 데이터 로드"""
    logger.info(f"days: {days}")
    
    db = ExcelRecommendationDB()
        
    interactions = db.get_user_item_interactions(days=days)
    
    if interactions.empty:
        logger.error(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
        raise ValueError(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
    
    logger.info(f"총 {len(interactions)}개의 상호작용 데이터 로드 완료")
    return interactions

def load_data(days: int, use_test_db: bool = True, interaction_weights: dict = None):
    """
    데이터 로드 및 전처리
    
    Args:
        days: 데이터 로드 기간
        use_test_db: 테스트 DB 사용 여부
        interaction_weights: 상호작용 타입별 가중치
        
    Returns:
        전처리된 상호작용 데이터프레임
    """
    logger.info("데이터 로드 및 전처리 중...")
    
    # 데이터 로드
    interactions_df = load_interactions(days=days, use_test_db=use_test_db)
    
    # 상호작용 타입별 통계 출력
    interaction_stats = interactions_df['interaction_type'].value_counts()
    logger.info(f"\n상호작용 타입별 통계:")
    for itype, count in interaction_stats.items():
        logger.info(f"- {itype}: {count}건")
    logger.info(f"총 {len(interactions_df)}개의 상호작용 데이터 로드 완료")
    
    # rating 필드 추가
    if interaction_weights:
        interactions_df['rating'] = interactions_df['interaction_type'].map(interaction_weights)
        # 0 값은 제외
        interactions_df = interactions_df[interactions_df['rating'] > 0]
    
    return interactions_df

def prepare_spark(app_name: str):
    """
    Spark 세션 준비
    
    Args:
        app_name: Spark 애플리케이션 이름
        
    Returns:
        생성된 SparkSession
    """
    logger.info("Spark 세션 준비 중...")
    
    # SparkSession 생성
    spark_session = SparkSingleton.get(app_name=app_name, log_level="ERROR")
    logger.info("SparkSession 생성 완료")
    
    return spark_session

def prepare_matrices(interactions_df, spark_session, model_config):
    """
    행렬 데이터 준비
    
    Args:
        interactions_df: 상호작용 데이터프레임
        spark_session: Spark 세션
        model_config: 모델 설정
        
    Returns:
        tuple: (init_model, matrix_data)
    """
    logger.info("행렬 데이터 준비 중...")
    
    from models.pyspark_als import PySparkALS
    
    # 샘플 크기 제한 가져오기 - 모델 파라미터에서 분리
    max_sample_size = model_config.get('max_sample_size', 300000)
    
    # 모델 파라미터만 추출 (max_sample_size 및 기타 설정 제외)
    model_params = {}
    for key, value in model_config.items():
        if key not in ['max_sample_size', 'checkpoint_dir'] and not isinstance(value, list):
            model_params[key] = value
    
    # 모델 생성 (초기화만)
    init_model = PySparkALS(**model_params)
    init_model.spark = spark_session  # 세션 직접 지정
    
    # 전체 데이터로 행렬 변환
    matrix_data = init_model.prepare_matrices(
        interactions_df,
        max_sample_size=max_sample_size,
        split_strategy='random'  # 항상 랜덤 분할 사용
    )
    logger.info("행렬 변환 완료")
    
    return init_model, matrix_data

def run_pyspark_als(params, interactions_df, matrix_data, spark_session=None):
    """
    PySpark ALS 모델을 실행하고 결과를 반환
    
    Args:
        params: 하이퍼파라미터 딕셔너리
        interactions_df: 전체 상호작용 데이터프레임
        matrix_data: (ratings_df, train_data, test_data) 튜플
        spark_session: 재사용할 SparkSession (없으면 새로 생성)
        
    Returns:
        dict: 평가 지표와 하이퍼파라미터를 담은 딕셔너리
    """
    from models.pyspark_als import PySparkALS
    
    # 결과 저장 딕셔너리에 하이퍼파라미터 추가
    result = params.copy()
    
    # 기본값으로 NaN 설정 - 실패 시 시각화에서 필터링 가능하도록
    result['test_rmse'] = np.nan
    result['test_huber_loss'] = np.nan
    result['validation_weighted_sum'] = np.nan
    
    try:
        # 모델 생성
        model = PySparkALS(**params)
        
        # 기존 SparkSession 사용 설정
        if spark_session is not None:
            model.spark = spark_session
        
        # 시작 시간 기록
        start_time = time.time()
        
        # 모델 학습
        model.train(interactions_df, matrix_data)
        
        # 학습 시간 기록
        train_time = time.time() - start_time
        result['train_time'] = train_time
        
        # 테스트 데이터로 평가
        ratings_df, train_data, test_data = matrix_data
        
        # 테스트 데이터 검사
        test_count = test_data.count()
        logger.info(f"테스트 데이터 크기: {test_count}개 row")
        
        if test_count == 0:
            logger.warning("테스트 데이터가 비어있습니다. 모든 메트릭을 NaN으로 설정합니다.")
            return result
        
        # 테스트 데이터에 대한 예측 생성
        try:
            predictions = model.transform(test_data)
            pred_count = predictions.count()
            logger.info(f"예측 결과 크기: {pred_count}개 row")
            
            if pred_count == 0:
                logger.warning("예측 결과가 비어있습니다. coldStartStrategy 문제일 수 있습니다.")
                return result
            
            # NaN 값을 가진 행 제거 (coldStartStrategy='nan' 때문에 발생)
            predictions = predictions.dropna(subset=['prediction'])
            filtered_pred_count = predictions.count()
            
            if filtered_pred_count == 0:
                logger.warning("NaN 예측을 제거한 후 예측 결과가 비어있습니다.")
                return result
            
            logger.info(f"NaN 예측 제거 후 결과 크기: {filtered_pred_count}개 row")
            
            # 테스트 손실 계산
            evaluator_rmse = RegressionEvaluator(
                metricName="rmse", 
                labelCol="rating",
                predictionCol="prediction"
            )
            
            try:
                test_rmse = evaluator_rmse.evaluate(predictions)
                result['test_rmse'] = test_rmse
            except Exception as e:
                logger.warning(f"RMSE 계산 중 오류: {str(e)}")
            
            # Huber Loss 계산 (지원되는 경우)
            try:
                test_huber_loss = calculate_huber_loss(predictions)
                result['test_huber_loss'] = test_huber_loss
            except Exception as e:
                logger.warning(f"Huber Loss 계산 중 오류: {str(e)}")
                
            # 검증 가중 합계 계산 - Spark 기반 메서드 직접 호출
            try:
                logger.info("Spark 기반 Top-N 추천 검증 시작")
                validation_weighted_sum = calculate_validation_weighted_sum_spark(
                    model=model.model,
                    test_data_spark=test_data, 
                    top_n=20,
                    logger=logger
                )
                result['validation_weighted_sum'] = validation_weighted_sum
            except Exception as e:
                logger.warning(f"Spark 기반 검증 실패: {str(e)}")
                
        except Exception as e:
            logger.error(f"예측 생성 또는 평가 중 오류: {str(e)}")
        
        # 메모리 해제 시간을 주기
        time.sleep(2)
        
        # 모델 정리 (SparkSession 유지)
        if spark_session is not None:
            # 외부에서 제공된 SparkSession은 종료하지 않고, 모델 객체만 정리
            model.model = None  # 모델 객체 참조 제거
            # 명시적으로 외부 세션임을 표시
            model._session_created_internally = False
        else:
            # 내부에서 생성한 SparkSession은 함께 정리
            model._session_created_internally = True
            model.cleanup()
        
        del model
        gc.collect()
        
    except Exception as e:
        logger.error(f"모델 실행 중 오류: {str(e)}")
        result['error'] = str(e)
        
    return result

def run_tuning(params_list, interactions_df, matrix_data, spark_session):
    """
    하이퍼파라미터 튜닝 루프 실행
    
    Args:
        params_list: 모델 파라미터 리스트
        interactions_df: 상호작용 데이터프레임
        matrix_data: 행렬 데이터
        spark_session: Spark 세션
        
    Returns:
        dict: 튜닝 결과 리스트
    """
    logger.info(f"하이퍼파라미터 튜닝 루프 시작 (총 {len(params_list)}개 조합)")
    
    # 튜닝 결과를 저장할 리스트
    tuning_results = []
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 모든 하이퍼파라미터 조합에 대해 모델 실행
    for i, params in enumerate(params_list):
        logger.info(f"\n=== 하이퍼파라미터 튜닝 진행 상황 ===")
        logger.info(f"진행: [{i+1}/{len(params_list)}] ({((i+1)/len(params_list))*100:.1f}%)")
        logger.info(f"현재 파라미터: max_iter={params.get('max_iter', 'N/A')}, reg_param={params.get('reg_param', 'N/A')}, rank={params.get('rank', 'N/A')}")
        
        # SparkSession 상태 확인
        if spark_session is not None:
            try:
                # 세션이 살아있는지 간단히 확인
                is_active = SparkSingleton.is_active(spark_session)
                if not is_active:
                    logger.warning("SparkSession이 활성 상태가 아닙니다. 새 세션을 생성합니다.")
                    spark_session = None
                    gc.collect()  # 메모리 정리
            except Exception as e:
                logger.warning(f"SparkSession 상태 확인 중 오류: {str(e)}. 새 세션을 생성합니다.")
                spark_session = None
                gc.collect()
        
        # 필요한 경우 새 SparkSession 생성
        if spark_session is None:
            try:
                logger.info("새 SparkSession 생성 중...")
                
                # 새 세션 생성
                spark_session = SparkSingleton.get(f"PySparkALS_Tuning_{timestamp}_{i}")
                
                # 파라미터에서 비모델 설정 제거 (max_sample_size 등)
                model_params = {k: v for k, v in params.items() if k not in ['max_sample_size', 'checkpoint_dir']}
                
                # 임시 모델 생성
                from models.pyspark_als import PySparkALS
                temp_model = PySparkALS(**model_params)
                temp_model.spark = spark_session  # 세션 직접 지정
                
                # 새로운 세션으로 matrix_data 다시 준비
                logger.info("새 세션으로 행렬 데이터 준비 중...")
                # max_sample_size 파라미터가 있으면 별도로 전달
                max_sample_size = params.get('max_sample_size', 300000)
                matrix_data = temp_model.prepare_matrices(
                    interactions_df,
                    max_sample_size=max_sample_size,
                    split_strategy='random'
                )
                
                # 임시 모델 정리 (세션은 유지)
                temp_model.model = None  # 모델 참조만 제거
                del temp_model
                gc.collect()
            except Exception as e:
                logger.error(f"새 SparkSession 생성 중 오류: {str(e)}")
                # 오류 발생 시 현재 조합 건너뛰기
                params['error'] = f"세션 생성 실패: {str(e)}"
                tuning_results.append(params)
                continue
        
        try:
            # 파라미터에서 비모델 설정 제거
            model_params = {k: v for k, v in params.items() if k not in ['max_sample_size', 'checkpoint_dir']}
            
            # 모델 학습 및 평가 - 기존 SparkSession 전달
            result = run_pyspark_als(model_params, interactions_df, matrix_data, spark_session)
            tuning_results.append(result)
            
            # 진행 상황 로깅
            if 'validation_weighted_sum' in result:
                logger.info(f"Validation Weighted Sum: {safe_format(result.get('validation_weighted_sum'))}")
            
            huber = result.get('test_huber_loss')
            rmse = result.get('test_rmse')
            
            if huber not in (None, np.nan):
                logger.info(f"Test Huber Loss: {safe_format(huber, '{:.6f}')}")
            elif rmse not in (None, np.nan):
                logger.info(f"Test RMSE: {safe_format(rmse, '{:.6f}')}")
            else:
                logger.info("Test metric unavailable")
            
            logger.info(f"학습 시간: {safe_format(result.get('train_time'), '{:.2f}')}초")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"조합 {params} 실행 실패: {error_msg}")
            
            # SparkContext 관련 오류 감지
            if "SparkContext" in error_msg or "Py4J" in error_msg:
                logger.warning("SparkContext 문제 감지: 새 세션을 생성합니다.")
                # 세션 리셋
                try:
                    if spark_session:
                        SparkSingleton.stop()
                except Exception:
                    pass  # 오류 무시
                spark_session = None
                gc.collect()
            
            # 실패한 경우에도 일부 정보 저장
            params['error'] = error_msg
            tuning_results.append(params)
            
        # 메모리 정리
        gc.collect()
    
    return tuning_results

def find_best_params(results_df, metric_name, higher_is_better=True):
    """
    특정 메트릭 기준으로 최적 파라미터 찾아 출력 및 반환
    
    Args:
        results_df: 결과 데이터프레임
        metric_name: 최적화 기준 메트릭 이름
        higher_is_better: 높은 값이 좋은지 여부
        
    Returns:
        dict: 최적 파라미터 정보
    """
    if metric_name not in results_df.columns:
        logger.info(f"{metric_name} 메트릭이 결과에 없습니다.")
        return None
        
    # NaN 값을 제외하고 최고값 찾기
    valid_df = results_df.dropna(subset=[metric_name])
    if valid_df.empty:
        logger.info(f"{metric_name}에 대한 유효한 결과가 없습니다.")
        return None
        
    # 최적값 찾기 (높은 값이 좋은지 또는 낮은 값이 좋은지에 따라)
    if higher_is_better:
        best_idx = valid_df[metric_name].idxmax()
    else:
        best_idx = valid_df[metric_name].idxmin()
        
    best_config = results_df.iloc[best_idx]
    best_value = best_config[metric_name]
    
    # 로그 출력
    logger.info(f"\n=== 최적의 하이퍼파라미터 ({metric_name} 기준) ===")
    logger.info(f"max_iter: {best_config.get('max_iter', 'N/A')}")
    logger.info(f"reg_param: {best_config.get('reg_param', 'N/A')}")
    logger.info(f"rank: {best_config.get('rank', 'N/A')}")
    logger.info(f"Best {metric_name}: {safe_format(best_value)}")
    
    # 최적 하이퍼파라미터 정보 반환
    return {
        "metric": metric_name,
        "value": float(best_value),
        "params": {
            "max_iter": int(best_config.get('max_iter', 0)),
            "reg_param": float(best_config.get('reg_param', 0)),
            "rank": int(best_config.get('rank', 0))
        }
    }

def save_results(tuning_results, output_dir, timestamp, model_config, days):
    """
    튜닝 결과 저장 및 시각화
    
    Args:
        tuning_results: 튜닝 결과 리스트
        output_dir: 출력 디렉토리
        timestamp: 실행 타임스탬프
        model_config: 모델 설정
        days: 데이터 로드 기간
        
    Returns:
        최적 파라미터 정보 딕셔너리
    """
    logger.info("튜닝 결과 저장 및 시각화 중...")
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(tuning_results)
    
    # 결과 파일 저장
    results_path = os.path.join(output_dir, f"tuning_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"튜닝 결과가 {results_path}에 저장되었습니다.")
    
    # 실행 설정 저장
    config_info = {
        "model_type": "pyspark_als",
        "days": days,
        "test_size": 0.2,
        "random_state": 42,
        "hyperparameters_tested": len(tuning_results),
        "data_size": len(tuning_results),
        "timestamp": timestamp,
        "parameter_space": {k: v for k, v in model_config.items() if isinstance(v, list)}
    }
    
    # 설정 정보 저장
    with open(os.path.join(output_dir, "config_info.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    # 결과 시각화
    visualize_results(results_df, output_dir)
    
    # 최적의 하이퍼파라미터 출력
    best_metrics = {}
    
    # 각 메트릭 기준으로 최적 파라미터 찾기
    vws_best = find_best_params(results_df, 'validation_weighted_sum', higher_is_better=True)
    if vws_best:
        best_metrics['validation_weighted_sum'] = vws_best
        
    huber_best = find_best_params(results_df, 'test_huber_loss', higher_is_better=False)
    if huber_best:
        best_metrics['test_huber_loss'] = huber_best
        
    rmse_best = find_best_params(results_df, 'test_rmse', higher_is_better=False)
    if rmse_best:
        best_metrics['test_rmse'] = rmse_best
    
    # 최적 파라미터 정보가 없는 경우
    if not best_metrics:
        logger.warning("유효한 평가 결과가 없어 최적 파라미터를 찾을 수 없습니다.")
    
    # 모든 최적 파라미터 JSON으로 저장
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)
    
    return best_metrics

def generate_param_combinations(model_config):
    """
    하이퍼파라미터 조합 생성
    
    Args:
        model_config: 모델 설정
        
    Returns:
        list: 모델 파라미터 리스트
    """
    # 튜닝할 파라미터와 고정 파라미터 분리
    tuning_params = {}
    fixed_params = {}
    non_model_params = {}
    
    for key, value in model_config.items():
        if isinstance(value, list):
            tuning_params[key] = value
        elif key in ['max_sample_size', 'checkpoint_dir']:
            # 비모델 파라미터는 별도로 저장
            non_model_params[key] = value
        else:
            fixed_params[key] = value
    
    # 하이퍼파라미터 조합 생성
    param_keys = list(tuning_params.keys())
    param_values = [tuning_params[key] for key in param_keys]
    
    all_params_list = []
    for params_tuple in itertools.product(*param_values):
        # 파라미터 딕셔너리 생성 - 튜닝 파라미터
        params = {
            param_keys[i]: params_tuple[i] 
            for i in range(len(param_keys))
        }
        
        # 고정 파라미터 추가
        for key, value in fixed_params.items():
            params[key] = value
        
        # 비모델 파라미터 추가 (prepare_matrices 호출 시 사용됨)
        for key, value in non_model_params.items():
            params[key] = value
        
        all_params_list.append(params)
    
    logger.info(f"총 {len(all_params_list)}개의 하이퍼파라미터 조합 생성")
    return all_params_list

def main():
    """하이퍼파라미터 튜닝 실행"""
    try:
        logger.info("PySpark ALS 하이퍼파라미터 튜닝 시작")
        
        # 1. 설정 로드
        days = TUNING_CONFIG['default_params']['days']
        base_output_dir = TUNING_CONFIG['default_params']['output_dir']
        model_config = TUNING_CONFIG['pyspark_als']
        
        # 현재 시간 기반 폴더명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_dir = f"pyspark_als_{timestamp}"
        output_dir = os.path.join(base_output_dir, process_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"결과 저장 폴더: {output_dir}")
        
        # 2. 데이터 로드
        interactions_df = load_data(
            days=days, 
            use_test_db=True, 
            interaction_weights=model_config.get('interaction_weights')
        )
        
        # 3. 하이퍼파라미터 조합 생성
        params_list = generate_param_combinations(model_config)
        
        # 4. Spark 세션 준비
        spark_session = prepare_spark(f"PySparkALS_Tuning_{timestamp}")
        
        # 5. 행렬 데이터 준비
        init_model, matrix_data = prepare_matrices(interactions_df, spark_session, model_config)
        
        # 메모리 관리를 위해 모델 객체 제거 (SparkSession은 유지)
        init_model.model = None  # 모델 참조만 제거
        del init_model
        gc.collect()
        
        # 6. 튜닝 루프 실행
        tuning_results = run_tuning(params_list, interactions_df, matrix_data, spark_session)
        
        # 7. 결과 저장 및 시각화
        best_metrics = save_results(tuning_results, output_dir, timestamp, model_config, days)
        
        logger.info("하이퍼파라미터 튜닝 완료")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise
    finally:
        # 모든 작업 완료 후 SparkSession 종료
        if 'spark_session' in locals() and spark_session:
            try:
                logger.info("SparkSession 종료 중...")
                SparkSingleton.stop()
                logger.info("SparkSession 종료 완료")
            except Exception as e:
                logger.warning(f"SparkSession 종료 중 오류: {e}")

if __name__ == "__main__":
    main() 