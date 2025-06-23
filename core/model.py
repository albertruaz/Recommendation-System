"""
PySpark ALS 모델 핵심 로직
"""

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
from typing import Tuple, Optional
from utils.spark_utils import SparkSingleton
from utils.logger import setup_logger


class ALSModel:
    """PySpark ALS 모델의 핵심 로직만 담당"""
    
    def __init__(self, 
                 max_iter: int = 10,
                 reg_param: float = 0.1,
                 rank: int = 10,
                 random_state: int = 42,
                 nonnegative: bool = True,
                 cold_start_strategy: str = "nan"):
        
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.random_state = random_state
        self.nonnegative = nonnegative
        self.cold_start_strategy = cold_start_strategy
        
        self.spark = None
        self.model = None
        self.logger = setup_logger('als_model')
    
    def init_spark(self):
        """SparkSession 초기화"""
        if self.spark is None:
            self.spark = SparkSingleton.get(
                app_name=f"ALS_Model", 
                log_level="ERROR"
            )
        return self.spark
    
    def train(self, train_df) -> None:
        """ALS 모델 학습"""
        self.init_spark()
        
        # Spark DataFrame으로 변환
        spark_df = self.spark.createDataFrame(
            train_df[['user_idx', 'item_idx', 'rating']]
        )
        
        # ALS 모델 설정
        als = ALS(
            maxIter=self.max_iter,
            regParam=self.reg_param,
            rank=self.rank,
            userCol="user_idx",
            itemCol="item_idx", 
            ratingCol="rating",
            implicitPrefs=False,
            nonnegative=self.nonnegative,
            coldStartStrategy=self.cold_start_strategy,
            seed=self.random_state
        )
        
        # 학습
        self.logger.info("ALS 모델 학습 시작")
        self.model = als.fit(spark_df)
        self.logger.info("ALS 모델 학습 완료")
    
    def predict(self, test_df) -> pd.DataFrame:
        """예측 생성"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # Spark DataFrame으로 변환
        spark_test_df = self.spark.createDataFrame(
            test_df[['user_idx', 'item_idx', 'rating']]
        )
        
        # 예측 생성
        predictions = self.model.transform(spark_test_df)
        
        # pandas DataFrame으로 변환
        result_df = predictions.toPandas()
        
        return result_df
    
    def get_factors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """사용자 및 아이템 잠재 요인 추출"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        user_factors = self.model.userFactors.toPandas()
        item_factors = self.model.itemFactors.toPandas()
        
        return user_factors, item_factors
    
    def cleanup(self):
        """리소스 정리"""
        self.model = None
        if self.spark:
            SparkSingleton.stop()
            self.spark = None 