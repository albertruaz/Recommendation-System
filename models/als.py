"""
ALS 기반 추천 시스템 모델

이 모듈은 PySpark ALS를 사용하여 사용자-상품 추천을 생성하는 클래스를 제공합니다.
"""

import logging
import pandas as pd
from typing import Optional

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS as SparkALS

class ALSRecommender:
    """ALS 기반 추천 시스템 클래스"""
    
    def __init__(
        self,
        max_iter: int = 15,
        reg_param: float = 0.1,
        rank: int = 10,
        cold_start_strategy: str = "drop"
    ):
        """
        Args:
            max_iter (int): 최대 반복 횟수
            reg_param (float): 정규화 파라미터
            rank (int): 잠재 요인 개수
            cold_start_strategy (str): 콜드 스타트 처리 전략
        """
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.cold_start_strategy = cold_start_strategy
        self.model = None
        self.spark = None
        self._initialize_spark()
    
    def _initialize_spark(self) -> None:
        """SparkSession 초기화"""
        import findspark
        findspark.init()
        
        self.spark = (SparkSession.builder
            .appName("ALS Recommendation System")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .getOrCreate())
        
        self.spark.sparkContext.setLogLevel("ERROR")
    
    def train(self, interactions_df: pd.DataFrame) -> float:
        """모델 학습
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터
            
        Returns:
            float: RMSE 점수
        """
        try:
            # DataFrame 생성
            ratings_df = self.spark.createDataFrame(
                interactions_df[["member_id", "product_id", "rating"]]
            )
            
            # 학습/테스트 데이터 분할
            train_data, test_data = ratings_df.randomSplit([0.8, 0.2])
            
            # ALS 모델 설정
            als = SparkALS(
                maxIter=self.max_iter,
                regParam=self.reg_param,
                rank=self.rank,
                userCol="member_id",
                itemCol="product_id",
                ratingCol="rating",
                coldStartStrategy=self.cold_start_strategy
            )
            
            # 모델 학습
            self.model = als.fit(train_data)
            
            # 성능 평가
            predictions = self.model.transform(test_data)
            evaluator = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction"
            )
            rmse = evaluator.evaluate(predictions)
            logging.info(f"Root-mean-square error (RMSE): {rmse:.2f}")
            
            return rmse
            
        except Exception as e:
            logging.error(f"모델 학습 실패: {str(e)}")
            raise Exception(f"모델 학습 오류: {str(e)}")
    
    def generate_recommendations(self, top_n: int = 300) -> pd.DataFrame:
        """전체 사용자에 대한 추천 생성
        
        Args:
            top_n (int): 각 사용자당 추천할 상품 수
            
        Returns:
            pd.DataFrame: 추천 결과 데이터프레임
        """
        if self.model is None:
            raise Exception("모델이 학습되지 않았습니다. train() 메서드를 먼저 실행하세요.")
        
        try:
            # 전체 사용자에 대한 추천 생성
            user_recs = self.model.recommendForAllUsers(top_n)
            
            # recommendations 컬럼을 평탄화
            flattened_recs = user_recs.select(
                "member_id",
                F.explode("recommendations").alias("rec")
            )
            
            # rec 컬럼에서 product_id와 rating 분리
            flattened_recs = flattened_recs.select(
                "member_id",
                flattened_recs["rec.product_id"].alias("product_id"),
                flattened_recs["rec.rating"].alias("predicted_rating")
            )
            
            # Pandas DataFrame으로 변환
            recommendations_df = flattened_recs.toPandas()
            return recommendations_df
            
        except Exception as e:
            logging.error(f"추천 생성 실패: {str(e)}")
            raise Exception(f"추천 생성 오류: {str(e)}")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
            self.model = None 