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

from utils.spark_utils import SparkSingleton, safe_format, calculate_huber_loss, apply_prediction_cap, evaluate_model, calculate_validation_weighted_sum_spark
from utils.logger import setup_logger

class PySparkALS:
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
                 split_test_data: bool = False,
                 nonnegative: bool = True,
                 cold_start_strategy: str = "nan"):
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.random_state = random_state
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
        self.nonnegative = nonnegative  # 음수 예측 방지 여부
        self.cold_start_strategy = cold_start_strategy  # cold start 처리 방식
        
        # BaseALS에서 가져온 속성들
        self.user_factors = None
        self.item_factors = None
        self.user2idx = {}
        self.idx2user = []
        self.item2idx = {}
        self.idx2item = []
        
        # 로거 설정
        self.logger = setup_logger('als')
    
    def _create_spark_session(self):
        """필요한 설정으로 SparkSession을 생성합니다"""
        # 1. SparkSingleton을 통해 세션 가져오기
        self.spark = SparkSingleton.get(app_name=f"PySparkALS_{int(time.time())}", log_level="ERROR")
        
        # 세션 내부 생성 여부 표시
        self._session_created_internally = True
        
        return self.spark
    
    def prepare_df(self, interactions_df: pd.DataFrame, train_df=None, max_sample_size=300000, split_strategy='random'):
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
        
        # 사용할 데이터 선택
        if train_df is not None:
            interactions_df = train_df.copy()
        
        # rating 컬럼 보장
        if 'rating' not in interactions_df.columns:
            interactions_df['rating'] = interactions_df['interaction_type'].map(self.interaction_weights)
        
        # 2) 매핑된 인덱스 컬럼 추가
        interactions_df['user_idx'] = interactions_df['member_id'].map(self.user2idx)
        interactions_df['item_idx'] = interactions_df['product_id'].map(self.item2idx)
        
        # 3) Spark DataFrame 생성 - user_idx/item_idx와 rating만 사용
        als_data = interactions_df[['user_idx', 'item_idx', 'rating']]
        
        self.logger.info("Spark DataFrame 생성 중...")
        
        # SparkSession 생성이 필요한 경우만 생성 (외부에서 이미 생성된 경우 건너뜀)
        if self.spark is None:
            self._create_spark_session()
        
        try:
            # 한 번에 Spark DataFrame 생성 (PySpark가 내부적으로 적절히 분할함)
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
    
    def train(self, prepare_df) -> None:
        """
        PySpark ALS 모델 학습
        
        Args:
            prepare_df: Spark DataFrame 형태의 학습 데이터
        """
        try:
            # 세션 상태 확인
            if self.spark is None or not SparkSingleton.is_active(self.spark):
                raise RuntimeError("SparkSession이 활성 상태가 아닙니다. 새 세션이 필요합니다.")
            
            # 학습 시작 로깅
            self.logger.info(f"\n=== PySpark ALS 학습 시작 ===")
            self.logger.info(f"파라미터: max_iter={self.max_iter}, reg_param={self.reg_param}, rank={self.rank}")
            
            # ALS 모델 설정 - 매핑된 인덱스 컬럼 사용
            als = ALS(
                maxIter=self.max_iter,
                regParam=self.reg_param,
                rank=self.rank,
                userCol="user_idx",  # 변경: 매핑된 인덱스 컬럼 사용
                itemCol="item_idx",  # 변경: 매핑된 인덱스 컬럼 사용
                ratingCol="rating",
                implicitPrefs=False,    # Explicit 모드
                nonnegative=self.nonnegative,
                coldStartStrategy=self.cold_start_strategy,
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
            
            # prepare_df 직접 학습
            self.model = als.fit(prepare_df)
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
                max_safe_samples = int(os.environ.get('MAX_SAFE_SAMPLES', 100000))
                
                # Pandas로 변환
                # 자동 파티션 감지 또는 SparkSession 설정에서 가져온 값 사용
                pdf_user_factors = user_factors_df.toPandas()
                pdf_item_factors = item_factors_df.toPandas()
                
                # user_id/item_id가 아닌 인덱스를 기준으로 사용
                n_users = len(self.user2idx)
                n_items = len(self.item2idx)
                n_factors = self.rank
                
                # 사용자와 상품 요인을 저장할 배열 초기화
                self.user_factors = np.zeros((n_users, n_factors))
                self.item_factors = np.zeros((n_items, n_factors))
                
                # 요인 저장 (인덱스 기준)
                for _, row in pdf_user_factors.iterrows():
                    idx = int(row['id'])  # 이미 user_idx가 저장되어 있음
                    if 0 <= idx < n_users:  # 인덱스 범위 검사
                        self.user_factors[idx] = np.array(row['features'])
                
                for _, row in pdf_item_factors.iterrows():
                    idx = int(row['id'])  # 이미 item_idx가 저장되어 있음
                    if 0 <= idx < n_items:  # 인덱스 범위 검사
                        self.item_factors[idx] = np.array(row['features'])
                
                # 정리: 캐시 해제 및 관련 변수 제거
                user_factors_df.unpersist()
                item_factors_df.unpersist()
                del pdf_user_factors
                del pdf_item_factors
                gc.collect()
                
                self.logger.info(f"잠재 요인 추출 완료: 사용자({n_users}), 상품({n_items}), 요인({n_factors})")
                
            except Exception as e:
                self.logger.error(f"잠재 요인 추출 중 오류: {str(e)}")
                # 오류 발생 시 기본 차원으로 요인 초기화 (추천이라도 할 수 있게)
                n_users = len(self.user2idx)
                n_items = len(self.item2idx)
                self.user_factors = np.zeros((n_users, self.rank))
                self.item_factors = np.zeros((n_items, self.rank))
                raise
        except Exception as e:
            self.logger.error(f"잠재 요인 추출 전체 오류: {str(e)}")
            raise
    
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
    
    def generate_recommendations(self, top_n: int = 300) -> pd.DataFrame:
        """전체 사용자에 대한 추천 생성
        
        Args:
            top_n (int): 각 사용자당 추천할 상품 수
            
        Returns:
            pd.DataFrame: 추천 결과 데이터프레임
        """
        if self.user_factors is None or self.item_factors is None:
            raise Exception("모델이 학습되지 않았습니다. train() 메서드를 먼저 실행하세요.")
        
        try:
            self.logger.info(f"전체 사용자에 대한 추천 생성 시작 (top_n={top_n})")
            results = []
            
            # 각 사용자에 대해 추천 생성
            for (user_idx, user_id) in self.idx2user.items():
            
                # 예측 점수 계산 (내적)
                scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
                
                # 상위 N개 아이템 선택
                top_item_indices = np.argsort(-scores)[:top_n]
                top_scores = scores[top_item_indices]
                
                # 결과 저장
                user_results = pd.DataFrame({
                    'member_id': user_id,
                    'product_id': [self.idx2item[idx] for idx in top_item_indices],
                    'predicted_rating': top_scores
                })
                results.append(user_results)
            
            # 모든 결과 합치기
            recommendations_df = pd.concat(results, ignore_index=True)
            self.logger.info(f"추천 생성 완료: {len(recommendations_df)}개의 추천")
            return recommendations_df
            
        except Exception as e:
            self.logger.error(f"추천 생성 오류: {str(e)}")
            raise

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
            
            # 인덱스 매핑 추가
            test_df['user_idx'] = test_df['member_id'].map(self.user2idx)
            test_df['item_idx'] = test_df['product_id'].map(self.item2idx)
            
            # 매핑 실패한 행 필터링
            valid_test = test_df.dropna(subset=['user_idx', 'item_idx']).copy()
            if len(valid_test) < len(test_df):
                self.logger.warning(f"{len(test_df) - len(valid_test)}개의 행이 매핑 실패로 제외됨")
            
            # Spark DataFrame으로 변환 - 매핑된 인덱스 사용
            test_spark_df = self.spark.createDataFrame(
                valid_test[['user_idx', 'item_idx', 'rating']]
            )
            
            # 예측 생성
            predictions = self.transform(test_spark_df)
            
            # 결과를 pandas DataFrame으로 변환
            result_df = predictions.select(
                'user_idx', 'item_idx', 'rating', 'prediction'
            ).toPandas()
            
            # 인덱스를 원래 ID로 변환
            result_df['member_id'] = result_df['user_idx'].map(self.idx2user)
            result_df['product_id'] = result_df['item_idx'].map(self.idx2item)
            
            # 원본 데이터와 병합
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

    def get_latent_factors(self):
        """
        학습된 모델의 사용자 및 아이템 잠재 요인을 반환합니다.
        
        Returns:
            tuple: (user_factors_df, item_factors_df) - 사용자 및 아이템 요인을 담은 DataFrame 튜플
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
            
        self.logger.info("잠재 요인 추출 중...")
        
        try:
            # 모델에서 사용자 및 아이템 요인 가져오기 (인덱스 기반)
            user_factors_df = self.model.userFactors.select("id", "features").toPandas()
            item_factors_df = self.model.itemFactors.select("id", "features").toPandas()
            
            # 인덱스를 원래 ID로 변환
            user_factors_df['member_id'] = user_factors_df['id'].apply(
                lambda x: self.idx2user.get(int(x)) if x in self.idx2user else None
            )
            item_factors_df['product_id'] = item_factors_df['id'].apply(
                lambda x: self.idx2item.get(int(x)) if x in self.idx2item else None
            )
            
            # 필요없는 열 제거
            user_factors_df = user_factors_df.drop(columns=['id']).dropna(subset=['member_id'])
            item_factors_df = item_factors_df.drop(columns=['id']).dropna(subset=['product_id'])
            
            # features 열의 벡터를 풀어서 개별 컬럼으로 변환
            for i in range(self.rank):
                user_factors_df[f'factor_{i}'] = user_factors_df['features'].apply(lambda x: x[i] if len(x) > i else None)
                item_factors_df[f'factor_{i}'] = item_factors_df['features'].apply(lambda x: x[i] if len(x) > i else None)
            
            # features 열 제거
            user_factors_df = user_factors_df.drop(columns=['features'])
            item_factors_df = item_factors_df.drop(columns=['features'])
            
            self.logger.info(f"잠재 요인 추출 완료: 사용자({len(user_factors_df)}명), 상품({len(item_factors_df)}개)")
            
            return user_factors_df, item_factors_df
            
        except Exception as e:
            self.logger.error(f"잠재 요인 추출 중 오류 발생: {str(e)}")
            # 오류 발생 시 빈 DataFrame 반환
            return pd.DataFrame(columns=['member_id']), pd.DataFrame(columns=['product_id'])

    def _get_interaction_weight(self, row: pd.Series) -> float:
        """상호작용 타입에 따른 가중치 반환
        
        Args:
            row (pd.Series): 상호작용 데이터 행
                필수 컬럼:
                - interaction_type: 상호작용 타입 (impression1, impression2, view1, view2, like, cart, purchase)
            
        Returns:
            float: 계산된 가중치
        """
        interaction_type = row["interaction_type"]
        if interaction_type in self.interaction_weights:
            return self.interaction_weights[interaction_type]
        return 1.0

    def _log_training_results(self, interactions_df: pd.DataFrame, top_k: int = 20) -> None:
        """학습 결과를 분석하고 로그로 출력합니다.
        
        Args:
            interactions_df (pd.DataFrame): 원본 상호작용 데이터
            top_k (int): 출력할 상위 결과 개수
        """
        try:
            # 원본 데이터에서 실제 가중치 정보 추출
            analysis_df = interactions_df.copy()
            analysis_df['actual_weight'] = analysis_df.apply(self._get_interaction_weight, axis=1)
            
            # 예측값 계산
            predictions = []
            for _, row in analysis_df.iterrows():
                user_idx = self.user2idx[row['member_id']]
                item_idx = self.item2idx[row['product_id']]
                pred_score = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                predictions.append({
                    'user_id': row['member_id'],
                    'item_id': row['product_id'],
                    'interaction_type': row['interaction_type'],
                    'actual_weight': row['actual_weight'],
                    'predicted_score': pred_score
                })
            
            # 결과를 DataFrame으로 변환하고 정렬
            result_df = pd.DataFrame(predictions)
            result_df = result_df.sort_values('predicted_score', ascending=False)
            
            # 출력 디렉토리 생성
            os.makedirs('output', exist_ok=True)
            
            # CSV 파일로 저장
            output_path = 'output/weight_analysis.csv'
            result_df.to_csv(output_path, index=False)
            
            # Matrix 정보 로깅
            total_cells = len(self.user2idx) * len(self.item2idx)
            filled_cells = len(interactions_df)
            sparsity = (total_cells - filled_cells) / total_cells * 100
            
            self.logger.info("\n=== Matrix 정보 ===")
            self.logger.info(f"Shape: ({len(self.user2idx)}, {len(self.item2idx)})")
            self.logger.info(f"총 셀 수: {total_cells:,}")
            self.logger.info(f"채워진 셀 수: {filled_cells:,}")
            self.logger.info(f"Sparsity: {sparsity:.2f}%")
            
            # 상위 K개 결과 로깅 (actual_weight > 0인 것들 중에서)
            self.logger.info("\n=== 상위 결과 가중치 비교 ===")
            header = "Predicted Score | Actual Weight | Type        | User ID | Item ID"
            self.logger.info(header)
            self.logger.info("-" * len(header))
            
            count = 0
            for _, row in result_df.iterrows():
                if count >= top_k:
                    break
                    
                if row['actual_weight'] > 0:
                    self.logger.info(
                        f"{row['predicted_score']:>14.2f} | "
                        f"{row['actual_weight']:>12.2f} | "
                        f"{row['interaction_type']:<10s} | "
                        f"{row['user_id']:>7d} | "
                        f"{row['item_id']:>7d}"
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
            raise 