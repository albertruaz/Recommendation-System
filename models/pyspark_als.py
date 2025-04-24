"""
PySpark ALS 추천 시스템 구현

이 모듈은 PySpark ALS 알고리즘을 이용한 추천 시스템을 구현합니다.
"""

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col, lit, when
from pyspark.sql.types import FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import json
import os
import gc
import time

from models.base_als import BaseALS

class PySparkALS(BaseALS):
    """PySpark ALS 기반 추천 시스템 (Explicit Feedback)"""
    
    def __init__(self,
                 max_iter: int = 15,
                 reg_param: float = 0.1,
                 rank: int = 10,
                 random_state: int = 42,
                 alpha: float = 1.0,
                 interaction_weights: dict = None,
                 max_prediction: float = 50.0,
                 huber_delta: float = 10.0):
        super().__init__(max_iter, reg_param, rank, random_state)
        self.interaction_weights = interaction_weights or {
            "impression1": 1.0,
            "impression2": 0.5,
            "view1": 5.0,
            "view2": 7.0,
            "like": 10.0,
            "cart": 15.0,
            "purchase": 20.0,
            "review": 20.0
        }
        self.alpha = alpha
        self.spark = None
        self.model = None
        self.max_prediction = max_prediction  # 예측값 상한
        self.huber_delta = huber_delta  # Huber Loss의 델타값
    
    def _apply_prediction_cap(self, predictions_df):
        """예측값에 상한(cap)을 적용합니다.
        
        Args:
            predictions_df: 예측값을 포함한 Spark DataFrame
            
        Returns:
            상한이 적용된 Spark DataFrame
        """
        # 클래스 속성을 로컬 변수로 복사하여 UDF에서 참조하지 않도록 함
        max_pred_value = float(self.max_prediction)
        
        # SQL 표현식을 사용하여 상한 적용 (UDF 대신)
        return predictions_df.withColumn(
            "prediction", 
            F.when(col("prediction") > max_pred_value, max_pred_value)
                     .otherwise(col("prediction"))
        )
    
    def _calculate_huber_loss(self, predictions_df):
        """Huber Loss를 계산합니다.
        
        Args:
            predictions_df: 예측값과 실제값을 포함한 Spark DataFrame
            
        Returns:
            float: 계산된 Huber Loss 값
        """
        try:
            # 로컬 변수로 복사
            delta = float(self.huber_delta)
            
            # SQL 표현식을 사용하여 Huber Loss 계산 (UDF 없이)
            result = predictions_df.withColumn(
                "residual", F.abs(col("rating") - col("prediction"))
            ).withColumn(
                "huber_error",
                F.when(col("residual") <= delta, 
                      0.5 * col("residual") * col("residual"))
                 .otherwise(delta * (col("residual") - 0.5 * delta))
            )
            
            # 결과를 캐싱하여 메모리 부족 방지
            result = result.cache()
            
            # 메모리 효율적인 방법으로 평균 계산
            huber_loss = result.select(F.avg("huber_error").alias("avg_huber")).first()["avg_huber"]
            
            # 캐시 해제
            result.unpersist()
            
            return huber_loss
        except Exception as e:
            self.logger.error(f"Huber Loss 계산 중 오류: {str(e)}")
            # 오류 발생 시 대체 값 반환
            return float('nan')
    
    def _create_spark_session(self):
        """필요한 설정으로 SparkSession을 생성합니다"""
        # 기존 세션이 있으면 정리
        if SparkSession._instantiatedSession:
            SparkSession._instantiatedSession.stop()
        
        # log4j.properties 파일 경로 설정
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log4j_path = os.path.join(current_dir, "log4j.properties")
        
        # 새 세션 설정 및 생성
        self.spark = (
            SparkSession.builder
            .appName("PySparkALS")
            # 메모리 설정 추가
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            # 메모리 부족 방지를 위한 설정
            .config("spark.memory.fraction", "0.7")
            .config("spark.memory.storageFraction", "0.3")
            # GC 설정 - 로그 출력 비활성화
            .config("spark.driver.extraJavaOptions", 
                   f"-XX:+UseG1GC -XX:+UseCompressedOops -XX:-PrintGCDetails -XX:-PrintGCTimeStamps -Dlog4j.configuration=file:{log4j_path}")
            # 임시 파일 위치 설정
            .config("spark.local.dir", "/tmp")
            # Shuffle 관련 설정
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.default.parallelism", "8")
            # UI 비활성화 - 포트 바인딩 경고 메시지 방지
            .config("spark.ui.enabled", "false")
            # 추가적인 로그 억제 설정
            .config("spark.ui.showConsoleProgress", "false") 
            .getOrCreate()
        )
        
        # 로그 레벨 설정 - ERROR로 변경하여 중요 메시지만 표시
        self.spark.sparkContext.setLogLevel("ERROR")
        
        return self.spark
    
    def prepare_matrices(self, interactions_df: pd.DataFrame):
        """상호작용 데이터를 PySpark DataFrame으로 변환"""
        # 1) 인덱스 매핑
        self._prepare_indices(interactions_df)
        
        # 2) rating 계산 (interaction_weights에서 바로 가져옵니다)
        interactions_df['rating'] = interactions_df['interaction_type'].map(self.interaction_weights)
        
        # 3) rating == 0 인 데이터만 제외 (impression1/2는 1.0, 0.5이므로 살아남습니다)
        filtered = interactions_df[interactions_df['rating'] != 0].copy()
        self.filtered_interactions_df = filtered
        
        # 4) Spark DataFrame 생성
        als_data = filtered[['member_id','product_id','rating']]
        
        # SparkSession 생성 (최적화 설정 적용)
        self._create_spark_session()
        
        # 데이터 로드 및 파티션 최적화
        ratings_df = self.spark.createDataFrame(als_data)
        
        # 데이터 크기에 따라 파티션 수 최적화
        if len(als_data) > 10000:
            # 대략 8개의 파티션으로 재분배 (성능 테스트 후 조정 가능)
            ratings_df = ratings_df.repartition(8)
        
        # 5) train/test split
        train_data, test_data = ratings_df.randomSplit([0.8,0.2], seed=self.random_state)
        self.logger.info(f"훈련 데이터(>0만): {train_data.count()}, 테스트 데이터(>0만): {test_data.count()}")
        
        # 명시적 캐싱으로 성능 향상 및 메모리 관리
        train_data = train_data.cache()
        test_data = test_data.cache()
        
        return ratings_df, train_data, test_data
    
    def train(self, interactions_df, matrix_data) -> None:
        """모델 학습 및 로그 분석"""
        try:
            _, train_data, test_data = matrix_data
            
            # 학습
            self.logger.info(f"\n=== 학습 시작 ===")
            self.logger.info(f"총 반복 횟수: {self.max_iter}")
            
            # ALS 모델 설정
            als = ALS(
                maxIter=self.max_iter,
                regParam=self.reg_param,
                rank=self.rank,
                userCol="member_id",
                itemCol="product_id",
                ratingCol="rating",
                implicitPrefs=False,    # Explicit 모드
                nonnegative=True,       # 음수 예측 방지
                coldStartStrategy="drop",
                seed=self.random_state
            )
            
            # 학습 시작 전 시간 기록
            start_time = time.time()
            
            # 모델 학습 (이 단계에서 여러 반복이 수행됨)
            self.model = als.fit(train_data)
            
            # 학습 완료 후 메시지 출력
            self.logger.info(f"\n=== 학습 완료 ===")
            
            # 메모리 정리
            gc.collect()
            
            # Train 데이터 평가 - 예측값 상한 적용
            train_preds = self.model.transform(train_data)
            train_preds = self._apply_prediction_cap(train_preds)
            train_preds = train_preds.cache()  # 캐싱하여 반복 계산 최적화
            
            # 기본 RMSE
            train_rmse = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction"
            ).evaluate(train_preds)
            self.logger.info(f"Train RMSE: {train_rmse:.4f}")
            
            # Huber Loss 계산
            train_huber_loss = self._calculate_huber_loss(train_preds)
            self.logger.info(f"Train Huber Loss (delta={self.huber_delta}): {train_huber_loss:.4f}")
            
            # 메모리 관리를 위해 캐시 해제
            train_preds.unpersist()
            gc.collect()  # 명시적 GC 호출
            
            # Test 데이터 평가 - 예측값 상한 적용
            test_preds = self.model.transform(test_data)
            test_preds = self._apply_prediction_cap(test_preds)
            test_preds = test_preds.cache()  # 캐싱하여 반복 계산 최적화
            
            # 기본 RMSE
            test_rmse = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction"
            ).evaluate(test_preds)
            self.logger.info(f"Test RMSE: {test_rmse:.4f}")
            
            # Huber Loss 계산
            test_huber_loss = self._calculate_huber_loss(test_preds)
            self.logger.info(f"Test Huber Loss (delta={self.huber_delta}): {test_huber_loss:.4f}")
            
            # 캐시 해제
            test_preds.unpersist()
            gc.collect()  # 명시적 GC 호출
            
            # 평가 지표 저장
            self.metrics = {
                'train_rmse': train_rmse,
                'train_huber_loss': train_huber_loss,
                'test_rmse': test_rmse,
                'test_huber_loss': test_huber_loss
            }
            
            # 잠재 요인 추출 및 변환
            # 변환 전에 필요한 잠재 요인만 가져와서 처리 (메모리 최적화)
            user_factors_df = self.model.userFactors.select("id", "features").toPandas()
            item_factors_df = self.model.itemFactors.select("id", "features").toPandas()
            
            # 더 이상 필요 없는 데이터 삭제
            train_data.unpersist()
            test_data.unpersist()
            gc.collect()  # 명시적 GC 호출
            
            # 사용자 및 아이템 ID 매핑
            user_id_mapping = {i: user_id for i, user_id in enumerate(self.idx2user)}
            item_id_mapping = {i: item_id for i, item_id in enumerate(self.idx2item)}
            
            # 모델 매개변수(잠재 요인) 저장
            self.user_factors = np.zeros((len(self.user2idx), self.rank))
            self.item_factors = np.zeros((len(self.item2idx), self.rank))
            
            # 사용자 요인 설정
            for _, row in user_factors_df.iterrows():
                user_id = row['id']
                if user_id in user_id_mapping:
                    original_user_id = user_id_mapping[user_id]
                    if original_user_id in self.user2idx:
                        user_idx = self.user2idx[original_user_id]
                        self.user_factors[user_idx] = np.array(row['features'])
            
            # 아이템 요인 설정
            for _, row in item_factors_df.iterrows():
                item_id = row['id']
                if item_id in item_id_mapping:
                    original_item_id = item_id_mapping[item_id]
                    if original_item_id in self.item2idx:
                        item_idx = self.item2idx[original_item_id]
                        self.item_factors[item_idx] = np.array(row['features'])
            
            # 메모리 절약을 위해 필요 없는 데이터 삭제
            del user_factors_df
            del item_factors_df
            gc.collect()  # 명시적 GC 호출
            
            # 간소화된 로그 분석 - 메모리 효율성을 위해 샘플링 사용
            self.logger.info(f"PySpark ALS 모델 학습 완료")
        
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            # 에러 상황에서도 지표는 반환
            self.metrics = {
                'train_rmse': float('nan'),
                'train_huber_loss': float('nan'),
                'test_rmse': float('nan'),
                'test_huber_loss': float('nan')
            }
            # 학습 실패 시에도 기본 요인 설정
            self.user_factors = np.zeros((len(self.user2idx), self.rank))
            self.item_factors = np.zeros((len(self.item2idx), self.rank))
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            super().cleanup()
            
            # 모델 정리
            self.model = None
            
            # Spark 세션 종료
            if self.spark is not None:
                self.spark.stop()
                self.spark = None
                
            # 메모리 정리
            gc.collect()
            
            self.logger.info("PySpark 리소스 정리 완료")
        except Exception as e:
            self.logger.warning(f"리소스 정리 중 오류 발생: {str(e)}")
            # 오류가 발생해도 SparkSession은 정리시도
            if SparkSession._instantiatedSession:
                try:
                    SparkSession._instantiatedSession.stop()
                except:
                    pass 
    
    def _prepare_indices(self, interactions_df: pd.DataFrame):
        """사용자 및 상품 ID를 정수 인덱스로 매핑"""
        # 고유한 사용자 및 상품 ID 추출
        unique_users = interactions_df['member_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        # 인덱스 매핑 생성
        self.user2idx = {user: i for i, user in enumerate(unique_users)}
        self.item2idx = {item: i for i, item in enumerate(unique_items)}
        
        # 역매핑도 저장 (인덱스 -> ID)
        self.idx2user = unique_users
        self.idx2item = unique_items
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        self.logger.info(f"사용자 수: {self.n_users}, 상품 수: {self.n_items}")
    
    def _prepare_indices_from_matrix_data(self, matrix_data):
        """미리 생성된 행렬 데이터에서 인덱스 매핑 정보 가져오기"""
        # matrix_data에는 full_df, train_data, test_data가 있음
        full_df = matrix_data[0]
        
        # 고유한 사용자 및 상품 ID 추출
        user_ids = full_df.select("member_id").distinct().rdd.flatMap(lambda x: x).collect()
        item_ids = full_df.select("product_id").distinct().rdd.flatMap(lambda x: x).collect()
        
        # 인덱스 매핑 생성
        self.user2idx = {user: i for i, user in enumerate(user_ids)}
        self.item2idx = {item: i for i, item in enumerate(item_ids)}
        
        # 역매핑도 저장 (인덱스 -> ID)
        self.idx2user = user_ids
        self.idx2item = item_ids
        
        self.n_users = len(user_ids)
        self.n_items = len(item_ids)
        
        self.logger.info(f"사용자 수: {self.n_users}, 상품 수: {self.n_items}")
        return 