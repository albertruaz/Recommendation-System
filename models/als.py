"""
ALS 기반 추천 시스템 모델

이 모듈은 Implicit ALS를 사용하여 사용자-상품 추천을 생성하는 클래스를 제공합니다.
상호작용 가중치는 config/rating_weights.py에 정의되어 있습니다.
"""

import sys
sys.path.append("./libs/implicit")

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
# from implicit.als import _als_least_squares
# from implicit.als import _als
# from implicit.cpu._als import least_squares
from implicit.cpu import _als

from typing import Tuple, Dict, List

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

class ALSRecommender:
    """ALS 기반 추천 시스템 클래스"""
    
    def __init__(self, max_iter: int = 15, reg_param: float = 0.1,
                 rank: int = 10, random_state: int = 42):
        """
        Args:
            max_iter (int): 최대 반복 횟수
            reg_param (float): 정규화 파라미터
            rank (int): 잠재 요인 개수
            random_state (int): 랜덤 시드
        """
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.random_state = random_state
        
        self.user_factors = None  # X matrix
        self.item_factors = None  # Y matrix
        self.user2idx = {}
        self.idx2user = []
        self.item2idx = {}
        self.idx2item = []
    
    def _calculate_preference(self, row: pd.Series) -> float:
        """상호작용 타입에 따른 선호도 점수 계산
        
        Args:
            row (pd.Series): 상호작용 데이터 행
                필수 컬럼:
                - interaction_type: 상호작용 타입 (view, like, cart 등)
                - view_type: view 타입 (1, 2, 3), view인 경우에만 사용
            
        Returns:
            float: 계산된 선호도 점수 (0~1 사이 값)
        """
        interaction_type = row["interaction_type"]
        view_type = f"view_type_{int(row['view_type'])}" if pd.notna(row["view_type"]) else None
        
        # config에서 정의된 가중치 가져오기
        weight = CONFIG["interaction_weights"].get(view_type or interaction_type, 0.0)
        
        # min-max 정규화로 0~1 사이 값으로 변환
        max_weight = max(CONFIG["interaction_weights"].values())
        min_weight = min(CONFIG["interaction_weights"].values())
        
        if max_weight == min_weight:
            return weight / max_weight  # 모든 가중치가 같은 경우
            
        p_ui = (weight - min_weight) / (max_weight - min_weight)
        return p_ui
    
    def _prepare_matrices(self, interactions_df: pd.DataFrame) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """상호작용 데이터를 sparse matrix로 변환
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터
            
        Returns:
            Tuple[sp.csr_matrix, sp.csr_matrix]: (선호도 행렬, 신뢰도 행렬)
        """
        # 1. 유니크 사용자/상품 인덱싱
        unique_users = sorted(interactions_df["member_id"].unique())
        unique_items = sorted(interactions_df["product_id"].unique())
        
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = unique_users
        self.item2idx = {m: i for i, m in enumerate(unique_items)}
        self.idx2item = unique_items
        
        # 2. 선호도 점수(p_ui) 계산
        interactions_df["p_ui"] = interactions_df.apply(self._calculate_preference, axis=1)
        
        # 3. (user, item) 쌍별로 집계
        aggregated_df = interactions_df.groupby(
            ["member_id", "product_id"]
        )["p_ui"].max().reset_index()  # 여러 상호작용이 있으면 최대값 사용
        
        # 4. Sparse Matrix 생성
        row = aggregated_df["member_id"].map(self.user2idx)
        col = aggregated_df["product_id"].map(self.item2idx)
        p_ui = aggregated_df["p_ui"].values
        
        # P matrix (선호도)
        P = sp.coo_matrix(
            (p_ui, (row, col)),
            shape=(len(self.user2idx), len(self.item2idx))
        ).tocsr()
        
        # C matrix (신뢰도)
        # p_ui가 0이 아닌 모든 값에 대해 c_ui = 1 부여
        c_ui = np.ones_like(p_ui)
        C = sp.coo_matrix(
            (c_ui, (row, col)),
            shape=(len(self.user2idx), len(self.item2idx))
        ).tocsr()
        
        return P, C
    
    def train(self, interactions_df: pd.DataFrame) -> None:
        """모델 학습
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터
                필수 컬럼:
                - member_id: 사용자 ID
                - product_id: 상품 ID
                - interaction_type: 상호작용 타입 (view, like, cart 등)
                - view_type: view 타입 (1, 2, 3), view인 경우에만 사용
                - interaction_count: 해당 상호작용의 발생 횟수
        """
        try:
            # 1. P(선호도)와 C(신뢰도) 행렬 준비
            P, C = self._prepare_matrices(interactions_df)
            print(f"[DEBUG] P.shape = {P.shape}, P.nnz = {P.nnz}, "
                f"P.data.dtype = {P.data.dtype}, "
                f"max(P.indices) = {P.indices.max() if P.nnz else 'N/A'}, "
                f"max(P.indptr) = {P.indptr.max()}")

            print(f"[DEBUG] C.shape = {C.shape}, C.nnz = {C.nnz}, "
                f"C.data.dtype = {C.data.dtype}, "
                f"max(C.indices) = {C.indices.max() if C.nnz else 'N/A'}, "
                f"max(C.indptr) = {C.indptr.max()}")
            
            # 2. 잠재 요인 행렬 초기화
            np.random.seed(self.random_state)
            X = np.random.rand(len(self.user2idx), self.rank).astype(np.float32)  # 사용자 행렬
            Y = np.random.rand(len(self.item2idx), self.rank).astype(np.float32)  # 아이템 행렬
            print(f"[DEBUG] X.shape = {X.shape}, X.dtype = {X.dtype}, "
                f"X.min()={X.min()}, X.max()={X.max()}")
            print(f"[DEBUG] Y.shape = {Y.shape}, Y.dtype = {Y.dtype}, "
                f"Y.min()={Y.min()}, Y.max()={Y.max()}")
            
            C.data = C.data.astype(np.float32)
            C.indices = C.indices.astype(np.int32)
            C.indptr = C.indptr.astype(np.int32)
            P.data = P.data.astype(np.float32)
            P.indices = P.indices.astype(np.int32)
            P.indptr = P.indptr.astype(np.int32)
            print("[DEBUG] After casting C:")
            print(f"  C.shape = {C.shape}, C.nnz = {C.nnz}, "
                f"C.data.dtype = {C.data.dtype}, "
                f"max(C.indices) = {C.indices.max() if C.nnz else 'N/A'}, "
                f"max(C.indptr) = {C.indptr.max()}")
            print("[DEBUG] After casting P:")
            print(f"  P.shape = {P.shape}, P.nnz = {P.nnz}, "
                f"P.data.dtype = {P.data.dtype}, "
                f"max(P.indices) = {P.indices.max() if P.nnz else 'N/A'}, "
                f"max(P.indptr) = {P.indptr.max()}")
            # 3. ALS 반복 학습
            print("\n=== ALS 학습 시작 ===")
            for iteration in range(self.max_iter):
                print(f"\n--- Iteration {iteration+1}/{self.max_iter} ---")
                # (1) 아이템 행렬 갱신: Ct = C.T
                Ct = C.T  # CSC 형태
                print("[DEBUG] Ct (C.T) info:")
                print(f"  Ct.shape = {Ct.shape}, Ct.nnz = {Ct.nnz}, "
                    f"Ct.data.dtype = {Ct.data.dtype}, "
                    f"max(Ct.indices) = {Ct.indices.max() if Ct.nnz else 'N/A'}, "
                    f"max(Ct.indptr) = {Ct.indptr.max()}")
                # 혹시 CSC가 float64로 바뀌는지 다시 캐스팅
                Ct.data = Ct.data.astype(np.float32)
                Ct.indices = Ct.indices.astype(np.int32)
                Ct.indptr = Ct.indptr.astype(np.int32)

                # YtY
                YtY = Y.T @ Y
                print(f"[DEBUG] YtY.shape = {YtY.shape}, YtY.dtype = {YtY.dtype}, "
                    f"YtY.min()={YtY.min()}, YtY.max()={YtY.max()}")

                print("[DEBUG] _als._least_squares for ITEM update...")
                _als._least_squares(
                    YtY.astype(np.float32),
                    Ct.indptr,
                    Ct.indices,
                    Ct.data,
                    Y,   # 업데이트할 행렬
                    X,   # 고정된 행렬
                    self.reg_param,
                    0
                )

                print(f"[DEBUG] After ITEM update: Y.min={Y.min()}, Y.max={Y.max()}")

                # (2) 사용자 행렬 갱신: XtX
                XtX = X.T @ X
                print(f"[DEBUG] XtX.shape = {XtX.shape}, XtX.dtype = {XtX.dtype}, "
                    f"XtX.min()={XtX.min()}, XtX.max()={XtX.max()}")

                print("[DEBUG] _als._least_squares for USER update...")
                _als._least_squares(
                    XtX.astype(np.float32),
                    C.indptr,
                    C.indices,
                    C.data,
                    X,   # 업데이트할 행렬
                    Y,   # 고정된 행렬
                    self.reg_param,
                    0
                )
                print(f"[DEBUG] After USER update: X.min={X.min()}, X.max={X.max()}")
            
            self.user_factors = X
            self.item_factors = Y
            print("=== ALS 학습 완료 ===")
            
        except Exception as e:
            raise Exception(f"모델 학습 오류: {str(e)}")
    
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
            results = []
            
            # 각 사용자에 대해 추천 생성
            for user_idx, user_id in enumerate(self.idx2user):
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
            return recommendations_df
            
        except Exception as e:
            raise Exception(f"추천 생성 오류: {str(e)}")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.user_factors = None
        self.item_factors = None
        self.user2idx = {}
        self.idx2user = []
        self.item2idx = {}
        self.idx2item = [] 