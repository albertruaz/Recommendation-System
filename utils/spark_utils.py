"""
Spark 관련 유틸리티 모듈

이 모듈은 Spark 세션 관리 및 다양한 유틸리티 함수를 제공합니다.
"""

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
import logging

class SparkSingleton:
    """Spark 세션 싱글톤 관리 클래스"""
    _session = None
    
    @staticmethod
    def get(app_name="RecoApp", log_level="ERROR"):
        """
        Spark 세션을 가져오거나 생성합니다 (싱글톤 패턴)
        
        Args:
            app_name: Spark 애플리케이션 이름
            log_level: 로그 레벨 (ERROR, WARN, INFO)
            
        Returns:
            SparkSession: 생성된 또는 기존의 SparkSession
        """
        # 이미 세션이 있고 활성 상태인 경우 재사용
        if SparkSingleton._session is not None:
            try:
                # 간단한 작업으로 세션 활성 상태 확인
                SparkSingleton._session.sparkContext.parallelize([1]).count()
                return SparkSingleton._session
            except Exception:
                # 오류 발생 시 세션이 비활성화된 것으로 간주
                SparkSingleton._session = None
        
        # 새 세션 생성
        SparkSingleton._session = (
            SparkSession.builder
            .appName(app_name)
            # 메모리 설정
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.memory.fraction", "0.7")
            .config("spark.memory.storageFraction", "0.3")
            # 네트워크 설정
            .config("spark.network.timeout", "1200s")
            .config("spark.driver.maxResultSize", "1g")
            .config("spark.rpc.message.maxSize", "256")
            # 성능 설정
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.default.parallelism", "4")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.adaptive.enabled", "true")
            # UI 설정
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        
        # 로그 레벨 설정
        SparkSingleton._session.sparkContext.setLogLevel(log_level)
        
        return SparkSingleton._session
    
    @staticmethod
    def is_active(session):
        """
        주어진 Spark 세션이 활성 상태인지 확인합니다.
        
        Args:
            session: 확인할 SparkSession 객체
            
        Returns:
            bool: 세션이 활성 상태이면 True, 아니면 False
        """
        if session is None:
            return False
            
        try:
            # 1. isStopped 메서드 먼저 시도 (새 버전)
            if hasattr(session.sparkContext, 'isStopped'):
                return not session.sparkContext.isStopped
                
            # 2. 간단한 RDD 작업 시도 (이전 버전)
            test_rdd = session.sparkContext.parallelize([1, 2, 3])
            count = test_rdd.count()
            return count == 3
        except Exception as e:
            logging.debug(f"세션 활성화 확인 중 오류: {str(e)}")
            return False
    
    @staticmethod
    def stop():
        """현재 Spark 세션을 안전하게 종료합니다."""
        if SparkSingleton._session is not None:
            try:
                SparkSingleton._session.stop()
                SparkSingleton._session = None
                logging.info("SparkSession 종료 완료")
            except Exception as e:
                logging.warning(f"SparkSession 종료 중 오류: {e}")

def safe_format(value, fmt="{:.4f}", na="N/A"):
    """
    안전한 포맷팅 함수
    
    None이나 NaN 값에 대해 포맷 문자열을 적용할 때 발생하는 오류를 방지합니다.
    
    Args:
        value: 포맷팅할 값
        fmt: 포맷 문자열 템플릿 (기본값: "{:.4f}")
        na: 값이 None이거나 NaN인 경우 반환할 문자열 (기본값: "N/A")
        
    Returns:
        str: 포맷팅된 문자열 또는 na 값
    """
    try:
        # None 체크
        if value is None:
            return na
            
        # NaN 체크 (float 타입만)
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return na
            
        # 포맷 적용
        return fmt.format(value)
    except Exception:
        # 어떤 오류가 발생하든 na 반환
        return na

def apply_prediction_cap(predictions_df, max_prediction=50.0):
    """
    예측값에 상한(cap)을 적용합니다.
    
    Args:
        predictions_df: 예측값을 포함한 Spark DataFrame
        max_prediction: 예측값 상한 (기본값: 50.0)
        
    Returns:
        상한이 적용된 Spark DataFrame
    """
    return predictions_df.withColumn(
        "prediction", 
        F.when(F.col("prediction") > max_prediction, max_prediction)
                .otherwise(F.col("prediction"))
    )

def calculate_huber_loss(predictions_df, delta=10.0):
    """
    Huber Loss를 계산합니다.
    
    Args:
        predictions_df: 예측값과 실제값을 포함한 Spark DataFrame
        delta: Huber Loss 델타 파라미터 (기본값: 10.0)
        
    Returns:
        float: 계산된 Huber Loss 값 또는 NaN
    """
    try:
        # 로컬 변수로 복사
        delta_value = float(delta)
        
        # SQL 표현식을 사용하여 Huber Loss 계산 (UDF 없이)
        result = predictions_df.withColumn(
            "residual", F.abs(F.col("rating") - F.col("prediction"))
        ).withColumn(
            "huber_error",
            F.when(F.col("residual") <= delta_value, 
                  0.5 * F.col("residual") * F.col("residual"))
                .otherwise(delta_value * (F.col("residual") - 0.5 * delta_value))
        )
        
        # 결과를 캐싱하여 메모리 부족 방지
        result = result.cache()
        
        # 메모리 효율적인 방법으로 평균 계산
        huber_loss_row = result.select(F.avg("huber_error").alias("avg_huber")).first()
        
        # None 값 명시적 체크
        if huber_loss_row is None or huber_loss_row["avg_huber"] is None:
            huber_loss = float('nan')
        else:
            huber_loss = huber_loss_row["avg_huber"]
        
        # 캐시 해제
        result.unpersist()
        
        return huber_loss
    except Exception as e:
        logging.error(f"Huber Loss 계산 중 오류: {str(e)}")
        # 오류 발생 시 대체 값 반환
        return float('nan')

def evaluate_model(model, train_data, test_data, huber_delta=10.0, max_prediction=50.0):
    """
    ALS 모델을 평가하여 다양한 메트릭을 계산합니다.
    
    Args:
        model: 학습된 ALS 모델
        train_data: 학습 데이터 (Spark DataFrame)
        test_data: 테스트 데이터 (Spark DataFrame)
        huber_delta: Huber Loss 델타 파라미터 (기본값: 10.0)
        max_prediction: 예측값 상한 (기본값: 50.0)
        
    Returns:
        dict: 계산된 메트릭 딕셔너리
    """
    metrics = {}
    
    try:
        # 1. 훈련 데이터 평가
        train_preds = model.transform(train_data)
        train_preds = apply_prediction_cap(train_preds, max_prediction)
        
        # RegressionEvaluator를 사용하여 RMSE 계산
        evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="rating", 
            predictionCol="prediction"
        )
        
        train_rmse = evaluator.evaluate(train_preds)
        metrics['train_rmse'] = train_rmse
        
        # 훈련 데이터에 대한 Huber Loss 계산
        train_huber_loss = calculate_huber_loss(train_preds, huber_delta)
        metrics['train_huber_loss'] = train_huber_loss
        
        # 자원 해제
        train_preds.unpersist(blocking=True)
        
        # 2. 테스트 데이터 평가
        test_preds = model.transform(test_data)
        test_preds = apply_prediction_cap(test_preds, max_prediction)
        
        # NaN 예측 값 제거
        test_preds = test_preds.dropna(subset=['prediction'])
        
        # 예측이 비어있지 않은 경우에만 평가
        if test_preds.count() > 0:
            test_rmse = evaluator.evaluate(test_preds)
            metrics['test_rmse'] = test_rmse
            
            # 테스트 데이터에 대한 Huber Loss 계산
            test_huber_loss = calculate_huber_loss(test_preds, huber_delta)
            metrics['test_huber_loss'] = test_huber_loss
        else:
            metrics['test_rmse'] = float('nan')
            metrics['test_huber_loss'] = float('nan')
        
        # 자원 해제
        test_preds.unpersist(blocking=True)
        
    except Exception as e:
        logging.error(f"모델 평가 중 오류: {str(e)}")
        metrics = {
            'train_rmse': float('nan'),
            'train_huber_loss': float('nan'),
            'test_rmse': float('nan'),
            'test_huber_loss': float('nan')
        }
    
    return metrics 

def calculate_validation_weighted_sum_spark(model, test_data_spark, top_n=20, logger=None):
    """
    Spark 기반 검증 데이터의 각 사용자에 대해 추천 품질 점수 계산 (메모리 효율적)

    이 함수는 다음 단계로 작동합니다:
    1. 테스트 데이터에서 고유 사용자 목록 추출
    2. 해당 사용자들에 대한 추천 생성 (Spark 내장 recommendForUserSubset 사용)
    3. 추천 결과를 explode하여 각 사용자-상품 쌍으로 변환
    4. 실제 테스트 데이터와 조인하여 추천이 적중한 항목 식별
    5. 순위 가중치와 상호작용 가중치를 적용하여 최종 점수 계산

    Args:
        model: 학습된 ALS 모델 
        test_data_spark: 테스트/검증 데이터 (Spark DataFrame)
        top_n: 각 사용자마다 고려할 추천 상품 수
        logger: 로깅용 로거 객체 (없으면 표준 로깅 사용)
        
    Returns:
        float: 검증 데이터에 대한 추천 품질 점수 (가중치 합)
    """
    # 로거가 제공되지 않은 경우 기본 로거 사용
    if logger is None:
        logger = logging

    # 1) 테스트 사용자 서브셋 생성
    logger.info("Spark 기반 검증 점수 계산 중...")
    users_spark = test_data_spark.select("member_id").distinct()
    user_count = users_spark.count()

    if user_count == 0:
        logger.warning("테스트 데이터에 사용자가 없습니다.")
        return 0.0
        
    logger.info(f"테스트 사용자 수: {user_count}명")

    # 2) 모델을 사용하여 추천 생성
    logger.info(f"사용자당 Top-{top_n} 추천 생성 중...")
    recs = model.recommendForUserSubset(users_spark, top_n)

    # 3) 추천 결과 explode (사용자-상품-순위 형태로 변환)
    # Window 함수 사용하여 명확하게 수정
    window_spec = Window.partitionBy("member_id").orderBy(F.desc("prediction"))

    exploded_recs = recs.selectExpr(
        "member_id", 
        "explode(recommendations) as rec"
    ).selectExpr(
        "member_id", 
        "rec.product_id as product_id", 
        "rec.rating as prediction"
    ).withColumn(
        "rank", F.row_number().over(window_spec)
    )

    # 4) 테스트 데이터와 조인하여 적중 항목 찾기
    # 테스트 데이터에서 필요한 컬럼만 선택
    truth = test_data_spark.select("member_id", "product_id", "rating")

    # 추천 결과와 실제 데이터 inner join (적중 항목만 남김)
    joined = exploded_recs.join(
        truth, 
        on=["member_id", "product_id"], 
        how="inner"
    )

    # 5) 순위 가중치 계산 및 적용
    weighted_hits = joined.withColumn(
        "rank_weight",  # 순위 가중치: 1/log2(rank+2)
        F.lit(1.0) / F.log2(F.col("rank") + 2)
    ).withColumn(
        "weighted_score",  # 최종 점수: rating(상호작용 가중치) * 순위 가중치
        F.col("rating") * F.col("rank_weight")
    )

    # 6) 통계 계산
    # 각 사용자별 히트 수 계산
    user_hits = weighted_hits.groupBy("member_id").agg(
        F.count("*").alias("hit_count"),
        F.sum("weighted_score").alias("user_score")
    )

    # 각 사용자별 테스트 아이템 수 계산
    user_test_items = truth.groupBy("member_id").agg(
        F.count("*").alias("test_item_count")
    )

    # 사용자별 히트율 및 정규화 점수 계산
    user_metrics = user_hits.join(
        user_test_items, 
        on="member_id", 
        how="inner"
    ).withColumn(
        "hit_ratio",  # 히트율: 히트 수 / 테스트 아이템 수
        F.col("hit_count") / F.col("test_item_count")
    ).withColumn(
        "normalized_score",  # 정규화 점수: 원래 점수 * (1 + 히트율)/2
        F.col("user_score") * (F.lit(1.0) + F.col("hit_ratio")) / F.lit(2.0)
    )

    # 7) 전체 통계 계산
    agg_results = user_metrics.agg(
        F.avg("hit_ratio").alias("avg_hit_ratio"),
        F.avg("normalized_score").alias("avg_score"),
        F.count("*").alias("hit_users"),
        F.sum("hit_count").alias("total_hits")
    ).first()

    # 결과 추출 - None 값 안전 처리
    avg_hit_ratio = agg_results["avg_hit_ratio"] if agg_results["avg_hit_ratio"] is not None else 0.0
    avg_score = agg_results["avg_score"] if agg_results["avg_score"] is not None else 0.0
    hit_users = agg_results["hit_users"]
    total_hits = agg_results["total_hits"]

    # 로그 출력 - 안전한 포맷팅 적용
    logger.info(
        f"[Spark] 히트 사용자: {hit_users}/{user_count}, "
        f"총 히트 수: {total_hits}, "
        f"평균 히트율: {safe_format(avg_hit_ratio)}, "
        f"평균 가중치 합: {safe_format(avg_score)}"
    )

    # 상세 로깅 (일부 사용자 샘플)
    try:
        # 히트율이 높은 상위 5명의 사용자 샘플링
        top_users = user_metrics.orderBy(F.desc("hit_ratio")).limit(5)
        top_users_sample = top_users.collect()
        
        if len(top_users_sample) > 0:
            logger.info("==== 상위 사용자 히트율 샘플 ====")
            for row in top_users_sample:
                logger.info(
                    f"사용자 {row['member_id']}: "
                    f"히트 {row['hit_count']}/{row['test_item_count']}, "
                    f"히트율: {safe_format(row['hit_ratio'], '{:.2f}')}, "
                    f"정규화 점수: {safe_format(row['normalized_score'])}"
                )
    except Exception as e:
        logger.warning(f"상세 로깅 중 오류: {str(e)}")

    return avg_score 