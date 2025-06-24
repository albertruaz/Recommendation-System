import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
from typing import Tuple, Optional, List
from utils.spark_utils import SparkSingleton
from utils.logger import setup_logger


class ALSModel:
    
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
        if self.spark is None:
            self.spark = SparkSingleton.get(
                app_name=f"ALS_Model", 
                log_level="ERROR"
            )
        return self.spark
    
    def train(self, train_df) -> None:
        self.init_spark()
        
        spark_df = self.spark.createDataFrame(
            train_df[['user_idx', 'item_idx', 'rating']]
        )
        
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
        
        self.logger.info("ALS 모델 학습 시작")
        self.model = als.fit(spark_df)
        self.logger.info("ALS 모델 학습 완료")
    
    def predict(self, test_df) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        spark_test_df = self.spark.createDataFrame(
            test_df[['user_idx', 'item_idx', 'rating']]
        )
        
        predictions = self.model.transform(spark_test_df)
        
        result_df = predictions.toPandas()
        
        return result_df
    
    def get_factors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        user_factors = self.model.userFactors.toPandas()
        item_factors = self.model.itemFactors.toPandas()
        
        return user_factors, item_factors
    
    def get_factors_optimized(self) -> Tuple[np.ndarray, np.ndarray, List, List]:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        self.logger.info("최적화된 factors 추출 시작 (Pandas 건너뛰기)")
        
        user_data = self.model.userFactors.collect()
        item_data = self.model.itemFactors.collect()
        
        user_data_sorted = sorted(user_data, key=lambda x: x.id)
        item_data_sorted = sorted(item_data, key=lambda x: x.id)
        
        user_ids = [row.id for row in user_data_sorted]
        item_ids = [row.id for row in item_data_sorted]
        user_matrix = np.array([list(row.features) for row in user_data_sorted])
        item_matrix = np.array([list(row.features) for row in item_data_sorted])
        
        self.logger.info(f"최적화된 factors 추출 완료: 사용자 {user_matrix.shape}, 아이템 {item_matrix.shape}")
        
        return user_matrix, item_matrix, user_ids, item_ids
    
    def cleanup(self):
        self.model = None
        if self.spark:
            SparkSingleton.cleanup()
            self.spark = None 