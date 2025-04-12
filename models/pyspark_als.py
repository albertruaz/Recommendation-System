"""
PySpark ALS 추천 시스템 구현

이 모듈은 PySpark ALS 알고리즘을 이용한 추천 시스템을 구현합니다.
"""

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import json

from models.base_als import BaseALS

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

class PySparkALS(BaseALS):
    """PySpark ALS 기반 추천 시스템"""
    
    def __init__(self, max_iter: int = 15, reg_param: float = 0.1,
                 rank: int = 10, random_state: int = 42, alpha: float = 1.0):
        """
        Args:
            max_iter (int): 최대 반복 횟수
            reg_param (float): 정규화 파라미터
            rank (int): 잠재 요인 개수
            random_state (int): 랜덤 시드
            alpha (float): 신뢰 가중치 파라미터
        """
        super().__init__(max_iter, reg_param, rank, random_state)
        self.alpha = alpha
        self.spark = None
        self.model = None
    
    def prepare_matrices(self, interactions_df: pd.DataFrame):
        """상호작용 데이터를 PySpark DataFrame으로 변환
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터프레임
            
        Returns:
            tuple: (Spark DataFrame, train_data, test_data)
        """
        # 인덱스 매핑 생성
        self._prepare_indices(interactions_df)
        
        # 가중치 계산
        interactions_df["rating"] = interactions_df.apply(self._get_interaction_weight, axis=1)
        
        # 필요한 컬럼만 선택하고 PySpark DataFrame 생성을 위한 준비
        als_data = interactions_df[["member_id", "product_id", "rating"]]
        
        # Spark 세션 생성
        self.spark = SparkSession.builder \
            .appName("PySparkALS") \
            .getOrCreate()
            
        # Pandas DataFrame을 Spark DataFrame으로 변환
        ratings_df = self.spark.createDataFrame(als_data)
        
        # 데이터를 훈련셋과 테스트셋으로 분할 (8:2)
        train_data, test_data = ratings_df.randomSplit([0.8, 0.2], seed=self.random_state)
        
        self.logger.info(f"훈련 데이터: {train_data.count()}, 테스트 데이터: {test_data.count()}")
        
        return (ratings_df, train_data, test_data)
    
    def train(self, interactions_df, matrix_data) -> None:
        """모델 학습
        
        Args:
            interactions_df: 상호작용 데이터프레임
            matrix_data: prepare_matrices에서 반환된 데이터 (Spark DataFrame, train_data, test_data)
        """
        try:
            self.logger.info("PySpark ALS 모델 학습 시작")
            
            # matrix_data에서 데이터 추출
            _, train_data, test_data = matrix_data
            
            # ALS 모델 초기화
            als = ALS(
                maxIter=self.max_iter,
                regParam=self.reg_param,
                rank=self.rank,
                userCol="member_id",
                itemCol="product_id",
                ratingCol="rating",
                coldStartStrategy="drop",
                seed=self.random_state
            )
            
            # 모델 학습
            self.model = als.fit(train_data)
            
            # RMSE로 성능 평가
            predictions = self.model.transform(test_data)
            evaluator = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction"
            )
            rmse = evaluator.evaluate(predictions)
            self.logger.info(f"Root-mean-square error (RMSE): {rmse:.4f}")
            
            # 잠재 요인 추출 및 변환
            user_factors_df = self.model.userFactors.toPandas()
            item_factors_df = self.model.itemFactors.toPandas()
            
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
            
            self.logger.info(f"PySpark ALS 모델 학습 완료")
            
            # 학습 결과 분석 및 로깅
            self._log_training_results(interactions_df)
            
        except Exception as e:
            self.logger.error(f"PySpark ALS 모델 학습 오류: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """리소스 정리"""
        super().cleanup()
        
        # Spark 세션 종료
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
        
        self.model = None
        self.logger.info("PySpark 리소스 정리 완료") 