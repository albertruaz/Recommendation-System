import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf, col, row_number, collect_list
from pyspark.sql.window import Window
from typing import Tuple, Optional, List, Dict
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
        
        self.logger.info("모델 학습")
        self.logger.info(f"반복 횟수: {self.max_iter}")
        self.logger.info(f"잠재 요인 수: {self.rank}")
        self.logger.info(f"정규화 계수: {self.reg_param}")
        self.model = als.fit(spark_df)
        self.logger.info("학습 완료")
    
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
        
        self.logger.info("행렬 분해 결과 추출")
        
        user_data = self.model.userFactors.collect()
        item_data = self.model.itemFactors.collect()
        
        user_data_sorted = sorted(user_data, key=lambda x: x.id)
        item_data_sorted = sorted(item_data, key=lambda x: x.id)
        
        user_ids = [row.id for row in user_data_sorted]
        item_ids = [row.id for row in item_data_sorted]
        user_matrix = np.array([list(row.features) for row in user_data_sorted])
        item_matrix = np.array([list(row.features) for row in item_data_sorted])
        
        self.logger.info(f"사용자 행렬\t\t: {user_matrix.shape}")
        self.logger.info(f"아이템 행렬\t\t: {item_matrix.shape}")
        
        return user_matrix, item_matrix, user_ids, item_ids
    
    def recommend_for_all_users(self, n: int, train_df: pd.DataFrame) -> Dict[int, List[int]]:
        """
        모든 사용자에 대해 상위 n개의 아이템을 추천합니다.
        핵심 로직: 학습 데이터 제외 + 예측 점수 기반 top-N 추출
        
        Args:
            n: 추천할 아이템 수
            train_df: 학습에 사용된 데이터프레임 (user_idx, item_idx 컬럼 포함)
            
        Returns:
            Dict[int, List[int]]: 사용자별 추천 아이템 딕셔너리
        """
        import heapq
        from typing import Set
        
        self.logger.info(f"모든 사용자에 대해 top-{n} 추천 생성")
        
        # 1) 팩터·ID 가져오기
        U, V, users, items = self.get_factors_optimized()  # U: (U×k), V: (I×k)

        # 2) train_df → seen[user] = {item,…}
        seen: Dict[int, Set[int]] = train_df.groupby('user_idx')['item_idx']\
                                            .apply(set).to_dict()

        # 3) item_id → V 행 인덱스 매핑
        pos = {itm: i for i, itm in enumerate(items)}

        recs: Dict[int, List[int]] = {}
        for ui, u_id in enumerate(users):
            scores = U[ui] @ V.T
            for itm in seen.get(u_id, ()):
                scores[pos[itm]] = -np.inf
            topk = heapq.nlargest(n, range(len(scores)), key=scores.__getitem__)
            recs[u_id] = [items[i] for i in topk]

        self.logger.info(f"추천 생성 완료: {len(recs):,}명의 사용자")
        return recs

    def cleanup(self):
        self.model = None
        if self.spark:
            SparkSingleton.cleanup()
            self.spark = None 