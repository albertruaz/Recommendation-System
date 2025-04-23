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
import os

from models.base_als import BaseALS

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

class PySparkALS(BaseALS):
    """PySpark ALS 기반 추천 시스템 (Explicit Feedback)"""

    def __init__(self,
                 max_iter: int = 15,
                 reg_param: float = 0.1,
                 rank: int = 10,
                 random_state: int = 42,
                 alpha: float = 1.0):
        super().__init__(max_iter, reg_param, rank, random_state)
        # 이제 config에 정의된 interaction_weights를 그대로 가져옵니다.
        self.interaction_weights = CONFIG['pyspark_als']['interaction_weights']
        self.alpha = alpha
        self.spark = None
        self.model = None
    
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
        self.spark = SparkSession.builder.appName("PySparkALS").getOrCreate()
        ratings_df = self.spark.createDataFrame(als_data)
        
        # 5) train/test split
        train_data, test_data = ratings_df.randomSplit([0.8,0.2], seed=self.random_state)
        self.logger.info(f"훈련 데이터(>0만): {train_data.count()}, 테스트 데이터(>0만): {test_data.count()}")
        return ratings_df, train_data, test_data
    
    def train(self, interactions_df, matrix_data) -> None:
        """모델 학습 및 로그 분석"""
        _, train_data, test_data = matrix_data
        
        # Explicit ALS 세팅
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
        
        # 학습 & 평가
        self.model = als.fit(train_data)
        
        # Train 데이터 평가
        train_preds = self.model.transform(train_data)
        train_rmse = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        ).evaluate(train_preds)
        self.logger.info(f"Train RMSE: {train_rmse:.4f}")
        
        # Test 데이터 평가
        test_preds = self.model.transform(test_data)
        test_rmse = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        ).evaluate(test_preds)
        self.logger.info(f"Test RMSE: {test_rmse:.4f}")
        
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
        
        # 로그 분석: filtered_interactions_df와 예측 결과를 merge
        # Train 데이터 분석
        train_pred_pdf = train_preds.select("member_id","product_id","prediction").toPandas()
        train_analysis_df = pd.merge(
            self.filtered_interactions_df,
            train_pred_pdf,
            on=["member_id","product_id"],
            how="inner"
        )
        self.logger.info("\n=== Train Data 분석 결과 ===")
        self._log_training_results(train_analysis_df)
        
        # Test 데이터 분석
        test_pred_pdf = test_preds.select("member_id","product_id","prediction").toPandas()
        test_analysis_df = pd.merge(
            self.filtered_interactions_df,
            test_pred_pdf,
            on=["member_id","product_id"],
            how="inner"
        )
        self.logger.info("\n=== Test Data 분석 결과 ===")
        self._log_training_results(test_analysis_df)
        
        self.logger.info(f"PySpark ALS 모델 학습 완료")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        super().cleanup()
        
        # Spark 세션 종료
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
        
        self.model = None
        self.logger.info("PySpark 리소스 정리 완료")
        
    def _log_training_results(self, analysis_df: pd.DataFrame, top_k: int = 20) -> None:
        """학습 결과를 분석하고 로그로 출력합니다.
        
        Args:
            analysis_df (pd.DataFrame): 예측 결과가 포함된 상호작용 데이터프레임
                (member_id, product_id, interaction_type, rating, prediction 컬럼 포함)
            top_k (int): 출력할 상위 결과 개수
        """
        try:
            # 예측값과 실제값 확인 - 이미 analysis_df에 포함되어 있음
            result_df = analysis_df.rename(columns={
                'rating': 'actual_weight',
                'prediction': 'predicted_score'
            })
            
            # 예측 점수 기준으로 정렬
            result_df = result_df.sort_values('predicted_score', ascending=False)
            
            # 출력 디렉토리 생성
            os.makedirs('output', exist_ok=True)
            
            # CSV 파일로 저장
            output_path = 'output/pyspark_weight_analysis.csv'
            result_df.to_csv(output_path, index=False)
            
            # Matrix 정보 로깅
            total_cells = len(self.user2idx) * len(self.item2idx)
            filled_cells = len(analysis_df)
            sparsity = (total_cells - filled_cells) / total_cells * 100
            
            self.logger.info("\n=== Matrix 정보 ===")
            self.logger.info(f"Shape: ({len(self.user2idx)}, {len(self.item2idx)})")
            self.logger.info(f"총 셀 수: {total_cells:,}")
            self.logger.info(f"채워진 셀 수: {filled_cells:,}")
            self.logger.info(f"Sparsity: {sparsity:.2f}%")
            
            # 상위 K개 결과 로깅
            self.logger.info("\n=== 상위 결과 가중치 비교 ===")
            header = "Predicted Score | Actual Weight | Type        | User ID | Item ID"
            self.logger.info(header)
            self.logger.info("-" * len(header))
            
            count = 0
            for _, row in result_df.iterrows():
                if count >= top_k:
                    break
                    
                self.logger.info(
                    f"{row['predicted_score']:>14.2f} | "
                    f"{row['actual_weight']:>12.2f} | "
                    f"{row['interaction_type']:<10s} | "
                    f"{row['member_id']:>7d} | "
                    f"{row['product_id']:>7d}"
                )
                count += 1
            
            # 상호작용 타입별 통계
            self.logger.info("\n=== 상호작용 타입별 통계 ===")
            type_stats = result_df.groupby('interaction_type').agg({
                'actual_weight': ['mean', 'min', 'max', 'count'],
                'predicted_score': ['mean', 'min', 'max']
            }).round(3)
            
            self.logger.info("\n" + str(type_stats))
            self.logger.info(f"\n전체 분석 결과가 {output_path}에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"학습 결과 분석 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc()) 