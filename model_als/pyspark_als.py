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

from model_als.base_als import BaseALS
from utils.spark_utils import SparkSingleton, safe_format, calculate_huber_loss, apply_prediction_cap, evaluate_model, calculate_validation_weighted_sum_spark

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
                 huber_delta: float = 10.0,
                 split_test_data: bool = False):
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
        self.split_test_data = split_test_data  # 테스트 데이터 분할 여부
        self._session_created_internally = False  # 세션이 내부에서 생성되었는지 추적
    
    def _create_spark_session(self):
        """필요한 설정으로 SparkSession을 생성합니다"""
        # 1. SparkSingleton을 통해 세션 가져오기
        self.spark = SparkSingleton.get(app_name=f"PySparkALS_{int(time.time())}", log_level="ERROR")
        
        # 세션 내부 생성 여부 표시
        self._session_created_internally = True
        
        return self.spark
    
    def prepare_matrices(self, interactions_df: pd.DataFrame, train_df=None, max_sample_size=300000, split_strategy='random'):
        """
        상호작용 데이터를 PySpark DataFrame으로 변환하고 훈련/테스트 데이터로 분할
        
        Args:
            interactions_df: 원본 상호작용 데이터프레임
            test_df: 테스트 데이터프레임 (split_test_data가 True인 경우 사용)
            max_sample_size: 최대 샘플 크기 (기본값: 30만)
            split_strategy: 분할 전략 ('random': 무작위 분할, 'user': 사용자 기반 분할)
            
        Returns:
            tuple: (ratings_df, train_data, test_data) 형태의 Spark DataFrame 튜플
        """
        # 1) 인덱스 매핑 (항상 전체 데이터에 대해 인덱스 생성)
        self._prepare_indices(interactions_df)
        
        if train_df is not None:
            interactions_df = train_df
        
        if 'rating' not in interactions_df.columns:
            interactions_df['rating'] = interactions_df['interaction_type'].map(self.interaction_weights)
        filtered = interactions_df[interactions_df['rating'] > 0].copy()

        # 데이터 양이 너무 많은 경우 샘플링
        if len(filtered) > max_sample_size:
            self.logger.info(f"데이터가 너무 많아 샘플링합니다: {len(filtered)}개 → {max_sample_size}개")
            filtered = filtered.sample(n=max_sample_size, random_state=self.random_state)
        
        self.filtered_interactions_df = filtered
        
        # 4) Spark DataFrame 생성 - 필요한 열만 선택하여 메모리 사용량 줄임
        als_data = filtered[['member_id','product_id','rating']]
        
        # SparkSession 생성이 필요한 경우만 생성 (외부에서 이미 생성된 경우 건너뜀)
        if self.spark is None:
            self._create_spark_session()
        
        try:
            # 한 번에 Spark DataFrame 생성 (PySpark가 내부적으로 적절히 분할함)
            self.logger.info("Spark DataFrame 생성 중...")
            ratings_df = self.spark.createDataFrame(als_data)
            max_partitions = 2  # 최대 파티션 수 제한
            ratings_df = ratings_df.repartition(max_partitions)
            return ratings_df
        
        except Exception as e:
            self.logger.error(f"데이터 변환 중 오류 발생: {str(e)}")
            if self.spark is not None and SparkSession._instantiatedSession:
                try:
                    SparkSingleton.stop()
                except:
                    pass
            raise
    
    def train(self, matrix_data) -> None:
        """
        PySpark ALS 모델 학습
        
        Args:
            matrix_data: Spark DataFrame 형태의 학습 데이터
        """
        try:
            # 세션 상태 확인
            if self.spark is None or not SparkSingleton.is_active(self.spark):
                raise RuntimeError("SparkSession이 활성 상태가 아닙니다. 새 세션이 필요합니다.")
            
            # 학습 시작 로깅
            self.logger.info(f"\n=== PySpark ALS 학습 시작 ===")
            self.logger.info(f"파라미터: max_iter={self.max_iter}, reg_param={self.reg_param}, rank={self.rank}")
            
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
                coldStartStrategy="nan", # 'drop' 대신 'nan'을 사용하여 완전히 빈 예측 방지
                seed=self.random_state,
                # 메모리 관리를 위한 체크포인팅 활성화 (높은 rank와 iter에서 중요)
                intermediateStorageLevel="MEMORY_AND_DISK",
                finalStorageLevel="MEMORY_AND_DISK"
            )
            
            # 학습 시작
            start_time = time.time()
            
            # 메모리 부족 위험이 높은 경우 (높은 rank 또는 많은 반복) 체크포인트 디렉토리 설정
            if self.rank >= 30 or self.max_iter >= 30:
                # 환경 변수나 config에서 checkpoint_dir 가져오기 (없으면 기본값 사용)
                config_checkpoint_dir = os.environ.get('SPARK_CHECKPOINT_DIR', '/tmp')
                checkpoint_dir = f"{config_checkpoint_dir}/spark_checkpoint_{int(time.time())}"
                self.logger.info(f"높은 rank({self.rank}) 또는 iter({self.max_iter}) 감지: 체크포인트 활성화 ({checkpoint_dir})")
                self.spark.sparkContext.setCheckpointDir(checkpoint_dir)
            
            # matrix_data로 직접 학습
            self.model = als.fit(matrix_data)
            train_time = time.time() - start_time
            self.logger.info(f"학습 완료 (소요 시간: {safe_format(train_time, '{:.2f}')}초)")
            gc.collect()
            self._extract_factors_safely()
            self.logger.info("=== 모델 학습 완료 ===\n")
            
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류: {str(e)}")
            if hasattr(self, 'user2idx') and hasattr(self, 'item2idx'):
                self.user_factors = np.zeros((len(self.user2idx), self.rank))
                self.item_factors = np.zeros((len(self.item2idx), self.rank))
            raise
            
    def _extract_factors_safely(self):
        """
        모델의 잠재 요인을 안전하게 추출하는 별도 메서드
        메모리 부족 위험이 높은 이 부분을 분리하여 실패해도 기본 지표는 유지되도록 함
        """
        try:
            # 필요한 사용자 및 상품 잠재 요인 추출
            self.logger.info("잠재 요인 추출 중...")
            
            # 모델 잠재 요인 (최소한의 정보만 추출)
            user_factors_df = self.model.userFactors.select("id", "features").cache()
            item_factors_df = self.model.itemFactors.select("id", "features").cache()
            
            # 3. 검증 가중치 계산을 위한 잠재 요인을 Numpy 배열로 변환
            try:
                # 최대 안전 샘플 수를 환경 변수나 설정 파일에서 가져오기
                import os
                import json
                
                # 환경 변수에서 max_safe_count 가져오기
                max_safe_count = int(os.environ.get('SPARK_MAX_SAFE_COUNT', 100000))
                
                # 설정 파일이 있으면 해당 파일에서 값 읽기
                try:
                    if os.path.exists('config/tuning_config.json'):
                        with open('config/tuning_config.json', 'r') as f:
                            config = json.load(f)
                        max_safe_count = config.get('pyspark_als', {}).get('max_safe_count', max_safe_count)
                except Exception as e:
                    self.logger.debug(f"설정 파일 로드 중 오류 무시: {e}")
                
                # 높은 메모리 사용 위험 감지 및 경고
                user_count = user_factors_df.count()
                item_count = item_factors_df.count()
                memory_estimate = (user_count + item_count) * self.rank * 4 / 1024 / 1024  # MB 단위 대략적 메모리 사용량
                
                if memory_estimate > 500:  # 500MB 이상이면 경고
                    self.logger.warning(f"잠재 요인 변환에 예상되는 메모리: ~{safe_format(memory_estimate, '{:.1f}')}MB (사용자: {user_count}, 상품: {item_count}, rank: {self.rank})")
                
                # 메모리가 너무 많이 필요한 경우 샘플링 고려
                if user_count > max_safe_count or item_count > max_safe_count:
                    self.logger.warning(f"과도한 잠재 요인 수 감지, 샘플링 사용")
                    if user_count > max_safe_count:
                        frac = max_safe_count / user_count
                        user_factors_df = user_factors_df.sample(False, frac, seed=42)
                    if item_count > max_safe_count:
                        frac = max_safe_count / item_count
                        item_factors_df = item_factors_df.sample(False, frac, seed=42)
                
                # pandas로 변환 (성능 평가에만 사용되므로 메모리에 로드)
                user_factors_pd = user_factors_df.toPandas()
                item_factors_pd = item_factors_df.toPandas()
                
                # NumPy 배열 초기화
                self.user_factors = np.zeros((len(self.user2idx), self.rank))
                self.item_factors = np.zeros((len(self.item2idx), self.rank))
                
                # 사용자 요인 설정
                for _, row in user_factors_pd.iterrows():
                    user_id = row['id']
                    if user_id in self.idx2user:
                        original_user_id = self.idx2user[user_id]
                        if original_user_id in self.user2idx:
                            user_idx = self.user2idx[original_user_id]
                            self.user_factors[user_idx] = np.array(row['features'])
                
                # 아이템 요인 설정
                for _, row in item_factors_pd.iterrows():
                    item_id = row['id']
                    if item_id in self.idx2item:
                        original_item_id = self.idx2item[item_id]
                        if original_item_id in self.item2idx:
                            item_idx = self.item2idx[original_item_id]
                            self.item_factors[item_idx] = np.array(row['features'])
                
                self.logger.info(f"잠재 요인 변환 완료: 사용자({self.user_factors.shape}), 상품({self.item_factors.shape})")
                
                # 메모리 정리
                del user_factors_pd, item_factors_pd
                
            except Exception as e:
                self.logger.error(f"잠재 요인 변환 중 오류: {str(e)}")
                # 변환 실패 시에도 빈 배열만 설정하고 계속 진행
                if hasattr(self, 'user2idx') and hasattr(self, 'item2idx'):
                    self.user_factors = np.zeros((len(self.user2idx), self.rank))
                    self.item_factors = np.zeros((len(self.item2idx), self.rank))
            
            # 캐시 해제
            user_factors_df.unpersist(blocking=True)
            item_factors_df.unpersist(blocking=True)
            
            # 메모리 정리
            gc.collect()
            
        except Exception as e:
            # 요인 추출 실패 시 기본 요인 설정 후 경고만 표시하고 진행
            self.logger.error(f"잠재 요인 추출 실패: {str(e)}")
            if hasattr(self, 'user2idx') and hasattr(self, 'item2idx'):
                self.user_factors = np.zeros((len(self.user2idx), self.rank))
                self.item_factors = np.zeros((len(self.item2idx), self.rank))
    
    def transform(self, test_data):
        """
        테스트 데이터에 대한 예측 수행
        
        Args:
            test_data: 변환할 테스트 데이터 (Spark DataFrame)
            
        Returns:
            Spark DataFrame: 예측이 추가된 DataFrame
        """
        if self.model is None:
            raise ValueError("모델이 아직 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        try:
            # 예측 수행
            predictions = self.model.transform(test_data)
            predictions = apply_prediction_cap(predictions, self.max_prediction)
            
            return predictions
        except Exception as e:
            self.logger.error(f"데이터 변환 중 오류 발생: {str(e)}")
            # 가능한 경우 원본 데이터 반환
            return test_data
        
    def _prepare_indices(self, interactions_df: pd.DataFrame):
        """
        사용자 및 상품 ID를 연속적인 인덱스로 매핑
        
        Args:
            interactions_df: 상호작용 데이터프레임
        """
        # 중복 없는 사용자 및 상품 ID 추출
        unique_users = interactions_df['member_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        # 인덱스 매핑 사전 생성
        self.user2idx = {user: i for i, user in enumerate(unique_users)}
        self.item2idx = {item: i for i, item in enumerate(unique_items)}
        
        # 역방향 매핑도 생성 (인덱스 → ID)
        self.idx2user = {i: user for user, i in self.user2idx.items()}
        self.idx2item = {i: item for item, i in self.item2idx.items()}
        
        # 로깅
        n_users = len(unique_users)
        n_items = len(unique_items)
        self.logger.info(f"총 사용자 수: {n_users}, 총 상품 수: {n_items}")
        
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            # 모델 메모리 해제
            if hasattr(self, 'model') and self.model is not None:
                # 모델 참조 제거
                self.model = None
            
            # Spark 세션 종료 - 외부에서 공유된 세션을 사용 중이면 종료하지 않음
            if hasattr(self, '_session_created_internally') and self._session_created_internally:
                if hasattr(self, 'spark') and self.spark is not None:
                    try:
                        self.logger.info("내부에서 생성한 SparkSession 종료 중")
                        SparkSingleton.stop()
                        self.spark = None
                    except Exception as e:
                        self.logger.warning(f"SparkSession 종료 중 오류: {str(e)}")
            else:
                self.logger.info("외부에서 제공된 SparkSession은 종료하지 않음")
            
            # 명시적 GC 호출
            gc.collect()
            
        except Exception as e:
            self.logger.warning(f"리소스 정리 중 오류: {str(e)}")
            # 정리 중 오류가 발생해도 계속 진행 
            pass 

    def test(self, test_df):
        """
        테스트 데이터에 대한 예측을 생성하고 결과를 반환합니다.
        
        Args:
            test_df: 테스트 데이터프레임 (pandas DataFrame)
            
        Returns:
            pandas DataFrame: 원본 테스트 데이터와 예측값이 추가된 데이터프레임
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
            
        self.logger.info(f"테스트 데이터({len(test_df)}개)에 대한 예측 생성 중...")
        
        try:
            # rating 열 추가 (필요한 경우)
            if 'rating' not in test_df.columns and 'interaction_type' in test_df.columns:
                test_df['rating'] = test_df['interaction_type'].map(self.interaction_weights)
            
            # Spark DataFrame으로 변환
            test_spark_df = self.spark.createDataFrame(
                test_df[['member_id', 'product_id', 'rating']]
            )
            
            # 예측 생성
            predictions = self.transform(test_spark_df)
            
            # 결과를 pandas DataFrame으로 변환
            result_df = predictions.select(
                'member_id', 'product_id', 'rating', 'prediction'
            ).toPandas()
            
            # 원본 데이터와 병합
            # 'prediction' 열만 유지하면서 원본 테스트 데이터프레임과 병합
            merged_df = pd.merge(
                test_df,
                result_df[['member_id', 'product_id', 'prediction']],
                on=['member_id', 'product_id'],
                how='left'
            )
            
            self.logger.info(f"테스트 데이터에 대한 예측 생성 완료. 유효한 예측: {len(merged_df.dropna(subset=['prediction']))}개")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"테스트 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 데이터 반환
            return test_df 