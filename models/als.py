"""
ALS 기반 추천 시스템 모델

이 모듈은 Implicit ALS를 사용하여 사용자-상품 추천을 생성하는 클래스를 제공합니다.
상호작용 가중치는 config/rating_weights.py에 정의되어 있습니다.
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from typing import Tuple, Dict, List

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

class ALSRecommender:
    """ALS 기반 추천 시스템 클래스"""
    
    def __init__(self, max_iter: int = 15, reg_param: float = 0.1,
                 rank: int = 10, cold_start_strategy: str = "drop"):
        """
        Args:
            max_iter (int): 최대 반복 횟수
            reg_param (float): 정규화 파라미터
            rank (int): 잠재 요인 개수
            cold_start_strategy (str): 콜드 스타트 처리 전략 (인터페이스 유지용)
        """
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.cold_start_strategy = cold_start_strategy
        
        self.model = None
        self.train_matrix = None
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
        view_type = f"view_type_{int(row['view_type'])}" if pd.notna(row["view_type"]) else None
        
        if view_type and view_type in CONFIG["interaction_weights"]:
            return CONFIG["interaction_weights"][view_type]
        elif interaction_type in CONFIG["interaction_weights"]:
            return CONFIG["interaction_weights"][interaction_type]
        return 1.0
    
    def _calculate_confidence(self, weight: float) -> float:
        """신뢰도 점수 계산
        
        Args:
            weight (float): 원본 상호작용 점수
            
        Returns:
            float: 계산된 신뢰도 점수
        """
        return 1.0 + CONFIG["alpha"] * weight
    
    def _prepare_interaction_matrix(self, interactions_df: pd.DataFrame) -> Tuple[sp.csr_matrix, pd.DataFrame]:
        """상호작용 데이터를 sparse matrix로 변환
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터
            
        Returns:
            Tuple[sp.csr_matrix, pd.DataFrame]: (상호작용 행렬, 통합된 상호작용 데이터)
        """
        # 1. 유니크 사용자/상품 인덱싱
        unique_users = sorted(interactions_df["member_id"].unique())
        unique_items = sorted(interactions_df["product_id"].unique())
        
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = unique_users
        self.item2idx = {m: i for i, m in enumerate(unique_items)}
        self.idx2item = unique_items
        
        # 2. 각 상호작용에 가중치 적용
        interactions_df["weight"] = interactions_df.apply(self._get_interaction_weight, axis=1)
        
        # 3. 사용자-상품별로 가중치 합산
        aggregated_df = interactions_df.groupby(
            ["member_id", "product_id"]
        )["weight"].sum().reset_index()
        
        # 4. Confidence 적용 및 sparse matrix 생성
        row = aggregated_df["member_id"].map(self.user2idx)
        col = aggregated_df["product_id"].map(self.item2idx)
        data = aggregated_df["weight"].apply(self._calculate_confidence)
        
        matrix = sp.coo_matrix(
            (data, (row, col)),
            shape=(len(self.user2idx), len(self.item2idx))
        ).tocsr()
        
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
            # 데이터 준비
            self.train_matrix, aggregated_df = self._prepare_interaction_matrix(interactions_df)
            
            # 모델 초기화 및 학습
            self.model = AlternatingLeastSquares(
                factors=self.rank,
                regularization=self.reg_param,
                iterations=self.max_iter
            )
            self.model.fit(self.train_matrix)
            
            # RMSE 계산
            test_interactions = aggregated_df.sample(frac=0.2, random_state=42)
            test_users = test_interactions["member_id"].map(self.user2idx).values
            test_items = test_interactions["product_id"].map(self.item2idx).values
            actual = test_interactions["weight"].values
            
            predicted = []
            for user_idx, item_idx in zip(test_users, test_items):
                user_factors = self.model.user_factors[user_idx]
                item_factors = self.model.item_factors[item_idx]
                pred = np.dot(user_factors, item_factors)
                predicted.append(pred)
            
            rmse = np.sqrt(np.mean((np.array(predicted) - actual) ** 2))
            return rmse
            
        except Exception as e:
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
            results = []
            
            # 각 사용자에 대해 추천 생성
            for user_idx, user_id in enumerate(self.idx2user):
                # train_matrix[user_idx]를 사용해 이미 소비한 아이템을 필터링
                user_items = self.train_matrix[user_idx]
                
                # 추천 생성
                item_ids, scores = self.model.recommend(
                    userid=user_idx,
                    user_items=user_items,
                    N=top_n,
                    filter_already_liked_items=True
                )
                
                # 결과 저장
                user_results = pd.DataFrame({
                    'member_id': user_id,
                    'product_id': [self.idx2item[idx] for idx in item_ids],
                    'predicted_rating': scores
                })
                results.append(user_results)
            
            # 모든 결과 합치기
            recommendations_df = pd.concat(results, ignore_index=True)
            return recommendations_df
            
        except Exception as e:
            raise Exception(f"추천 생성 오류: {str(e)}")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.model = None
        self.train_matrix = None
        self.user2idx = {}
        self.idx2user = []
        self.item2idx = {}
        self.idx2item = [] 