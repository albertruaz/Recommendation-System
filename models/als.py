"""
ALS 기반 추천 시스템 모델

이 모듈은 Implicit ALS를 사용하여 사용자-상품 추천을 생성하는 클래스를 제공합니다.
상호작용 가중치는 config/rating_weights.py에 정의되어 있습니다.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from config.rating_weights import INTERACTION_WEIGHTS, ALPHA, MIN_INTERACTIONS

class ALSRecommender:
    """ALS 기반 추천 시스템 클래스"""
    
    def __init__(
        self,
        max_iter: int = 15,
        reg_param: float = 0.1,
        rank: int = 10,
        cold_start_strategy: str = "drop",  # 인터페이스 유지를 위해 남겨둠
        interaction_weights: Optional[Dict[str, float]] = None,
        min_interactions: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            max_iter (int): 최대 반복 횟수
            reg_param (float): 정규화 파라미터
            rank (int): 잠재 요인 개수
            cold_start_strategy (str): 콜드 스타트 처리 전략 (인터페이스 유지용)
            interaction_weights (Dict[str, float], optional): 상호작용별 가중치
            min_interactions (Dict[str, int], optional): 상호작용별 최소 횟수
        """
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.cold_start_strategy = cold_start_strategy
        self.interaction_weights = interaction_weights or INTERACTION_WEIGHTS
        self.min_interactions = min_interactions or MIN_INTERACTIONS
        self.model = None
        self.train_matrix = None  # 학습에 사용된 전체 사용자-아이템 행렬
        
        # user, item 인덱싱을 위한 매핑
        self.user2idx = {}
        self.idx2user = []
        self.item2idx = {}
        self.idx2item = []
    
    def _get_interaction_weight(self, row: pd.Series) -> float:
        """상호작용 타입에 따른 가중치 반환
        
        Args:
            row (pd.Series): 상호작용 데이터 행
                필수 컬럼:
                - interaction_type: 상호작용 타입 (view, like, cart 등)
                - view_type: view 타입 (1, 2, 3), view인 경우에만 사용
                - interaction_count: 해당 상호작용의 발생 횟수
            
        Returns:
            float: 계산된 가중치 (상호작용 횟수 * 기본 가중치)
        """
        interaction_type = row["interaction_type"]
        interaction_count = row["interaction_count"]
        
        # 최소 상호작용 횟수 체크
        if interaction_count < self.min_interactions.get(interaction_type, 1):
            return 0.0
            
        if interaction_type == "view":
            view_type = int(row["view_type"])  # float를 int로 변환
            base_weight = self.interaction_weights[f"view_type_{view_type}"]
        else:
            base_weight = self.interaction_weights[interaction_type]
            
        return base_weight * interaction_count
    
    def _calculate_confidence(self, rating: float) -> float:
        """신뢰도 점수 계산
        
        Args:
            rating (float): 원본 상호작용 점수
            
        Returns:
            float: 계산된 신뢰도 점수
        """
        return 1.0 + ALPHA * rating
    
    def _prepare_interaction_matrix(self, interactions_df: pd.DataFrame) -> Tuple[sp.csr_matrix, pd.DataFrame]:
        """상호작용 데이터를 sparse matrix로 변환
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터
            
        Returns:
            Tuple[sp.csr_matrix, pd.DataFrame]: (상호작용 행렬, 통합된 상호작용 데이터)
        """
        # 0. NaN 값 처리 및 데이터 타입 변환
        print("\n[1] 데이터 전처리 시작")
        print("전처리 전 데이터:")
        print(interactions_df[['member_id', 'product_id', 'interaction_type', 'interaction_count', 'view_type']].head())
        
        # NaN 값이 있는 행 제거
        interactions_df = interactions_df.dropna(subset=['member_id', 'product_id'])
        
        # member_id와 product_id를 정수형으로 변환
        interactions_df['member_id'] = interactions_df['member_id'].astype(int)
        interactions_df['product_id'] = interactions_df['product_id'].astype(int)
        
        print("\n전처리 후 데이터:")
        print(interactions_df[['member_id', 'product_id', 'interaction_type', 'interaction_count', 'view_type']].head())
        print(f"\n총 레코드 수: {len(interactions_df)}")
        print(f"유니크 사용자 수: {interactions_df['member_id'].nunique()}")
        print(f"유니크 상품 수: {interactions_df['product_id'].nunique()}")
        print("\n상호작용 타입별 개수:")
        print(interactions_df['interaction_type'].value_counts())
        
        # 1. 유니크 사용자/상품 인덱싱
        unique_users = sorted(interactions_df["member_id"].unique())  # 정렬하여 인덱스 일관성 유지
        unique_items = sorted(interactions_df["product_id"].unique())
        
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = unique_users
        self.item2idx = {m: i for i, m in enumerate(unique_items)}
        self.idx2item = unique_items
        
        # 2. 각 상호작용에 가중치 적용
        interactions_df["weight"] = interactions_df.apply(self._get_interaction_weight, axis=1)
        
        print("\n[2] 가중치 계산 결과 확인:")
        print(interactions_df[['member_id', 'product_id', 'interaction_type', 'interaction_count', 'view_type', 'weight']].head())
        print("\n가중치 통계:")
        print(interactions_df['weight'].describe())
        print("\n0이 아닌 가중치 수:", (interactions_df['weight'] > 0).sum())
        
        # 3. 사용자-상품별로 가중치 합산
        aggregated_df = interactions_df.groupby(
            ["member_id", "product_id"]
        )["weight"].sum().reset_index()
        
        print("\n[3] 사용자-상품별 가중치 합산 결과:")
        print(aggregated_df.head())
        print("\n합산된 가중치 통계:")
        print(aggregated_df['weight'].describe())
        print("\n0이 아닌 합산 가중치 수:", (aggregated_df['weight'] > 0).sum())
        
        # 4. Confidence 적용 및 sparse matrix 생성
        row = aggregated_df["member_id"].map(self.user2idx)
        col = aggregated_df["product_id"].map(self.item2idx)
        data = aggregated_df["weight"].apply(self._calculate_confidence)
        
        matrix = sp.coo_matrix(
            (data, (row, col)),
            shape=(len(self.user2idx), len(self.item2idx))
        ).tocsr()
        
        print("\n[4] Sparse Matrix 정보:")
        print(f"Shape: {matrix.shape}")
        print(f"Non-zero entries: {matrix.nnz}")
        print(f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
        print("\nConfidence 값 샘플 (처음 10개):")
        print(data[:10])
        
        return matrix, aggregated_df
    
    def train(self, interactions_df: pd.DataFrame) -> float:
        """모델 학습
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터
                필수 컬럼:
                - member_id: 사용자 ID
                - product_id: 상품 ID
                - interaction_type: 상호작용 타입 (view, like, cart 등)
                - view_type: view 타입 (1, 2, 3), view인 경우에만 사용
                - interaction_count: 해당 상호작용의 발생 횟수
            
        Returns:
            float: RMSE 점수
        """
        try:
            print("\n=== ALS 모델 학습 시작 ===")
            
            # 1. 데이터 전처리
            train_matrix, aggregated_df = self._prepare_interaction_matrix(interactions_df)
            
            # 학습에 사용된 행렬 저장
            self.train_matrix = train_matrix
            
            # 2. train/test split (8:2)
            test_mask = np.random.rand(len(aggregated_df)) < 0.2
            test_df = aggregated_df[test_mask]
            
            print(f"\n[5] Train/Test Split 정보:")
            print(f"전체 데이터 수: {len(aggregated_df)}")
            print(f"테스트 데이터 수: {len(test_df)}")
            
            # 3. 모델 학습
            self.model = AlternatingLeastSquares(
                factors=self.rank,
                regularization=self.reg_param,
                iterations=self.max_iter
            )
            
            print("\n모델 학습 시작...")
            self.model.fit(train_matrix)
            print("모델 학습 완료!")
            
            print("\n[6] 학습된 Factors 정보:")
            print(f"User factors shape: {self.model.user_factors.shape}")
            print(f"Item factors shape: {self.model.item_factors.shape}")
            print("\nUser factors 샘플 (처음 5개):")
            print(self.model.user_factors[:5])
            print("\nItem factors 샘플 (처음 5개):")
            print(self.model.item_factors[:5])
            
            # 4. RMSE 계산
            test_users = test_df["member_id"].map(self.user2idx)
            test_items = test_df["product_id"].map(self.item2idx)
            test_ratings = test_df["weight"].values
            
            user_factors = self.model.user_factors[test_users]
            item_factors = self.model.item_factors[test_items]
            pred = np.sum(user_factors * item_factors, axis=1)
            
            rmse = np.sqrt(np.mean((test_ratings - pred) ** 2))
            print(f"\n[7] RMSE 계산 결과: {rmse:.4f}")
            print(f"테스트 예측값 통계:")
            print(f"평균: {np.mean(pred):.4f}")
            print(f"표준편차: {np.std(pred):.4f}")
            print(f"최소값: {np.min(pred):.4f}")
            print(f"최대값: {np.max(pred):.4f}")
            
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
        if self.train_matrix is None:
            raise Exception("train_matrix가 없습니다. train() 메서드에서 self.train_matrix에 저장했는지 확인하세요.")
        
        try:
            print("\n=== 추천 생성 시작 ===")
            results = []
            
            print(f"\n[8] 추천 생성 정보:")
            print(f"전체 사용자 수: {len(self.user2idx)}")
            print(f"전체 상품 수: {len(self.item2idx)}")
            print(f"사용자당 추천 수: {top_n}")
            
            # 각 사용자에 대해 추천 생성
            for user_idx, user_id in enumerate(self.idx2user):
                # train_matrix[user_idx]를 사용해 이미 소비한 아이템을 필터링
                user_items = self.train_matrix[user_idx]
                
                # 현재 사용자의 상호작용 정보 출력
                if user_idx < 5:  # 처음 5명의 사용자에 대해서만 출력
                    print(f"\n사용자 {user_id} (idx: {user_idx})의 상호작용:")
                    print(f"0이 아닌 상호작용 수: {user_items.nnz}")
                    if user_items.nnz > 0:
                        print("상호작용 값 샘플:", user_items.data[:5])
                
                # 추천 생성
                item_ids, scores = self.model.recommend(
                    userid=user_idx,
                    user_items=user_items,
                    N=top_n,
                    filter_already_liked_items=True
                )
                
                # 추천 결과 샘플 출력 (처음 5명의 사용자에 대해서만)
                if user_idx < 5:
                    print(f"\n사용자 {user_id}의 추천 결과 (상위 5개):")
                    for item_idx, score in zip(item_ids[:5], scores[:5]):
                        print(f"상품 {self.idx2item[item_idx]}: {score:.4f}")
                
                # 결과 저장
                user_results = pd.DataFrame({
                    'member_id': user_id,
                    'product_id': [self.idx2item[idx] for idx in item_ids],
                    'predicted_rating': scores
                })
                results.append(user_results)
            
            # 모든 결과 합치기
            recommendations_df = pd.concat(results, ignore_index=True)
            
            print("\n[9] 최종 추천 결과 통계:")
            print(recommendations_df['predicted_rating'].describe())
            print("\n0이 아닌 추천 점수 수:", (recommendations_df['predicted_rating'] > 0).sum())
            print(f"전체 추천 수: {len(recommendations_df)}")
            
            return recommendations_df
            
        except Exception as e:
            logging.error(f"추천 생성 실패: {str(e)}")
            raise Exception(f"추천 생성 오류: {str(e)}")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.model = None
        self.train_matrix = None  # train_matrix도 정리
        self.user2idx = {}
        self.item2idx = {}
        self.idx2user = []
        self.idx2item = [] 