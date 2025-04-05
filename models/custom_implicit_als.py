"""
ALS 기반 추천 시스템 모델

이 모듈은 명시적 선호도를 사용하는 ALS를 구현합니다.
"""
import os
import sys
sys.path.append("./libs/implicit")

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.cpu import _als
from utils.logger import setup_logger

import time



##
import numpy as np
from scipy.sparse import csr_matrix

def least_squares_python(Cui: csr_matrix, 
                         X: np.ndarray, 
                         Y: np.ndarray, 
                         regularization: float):
    """
    - Cui: (n_users, n_items) CSR matrix
           data[u, i] = confidence (alpha * rating 등)
           'nonzero'라면 implicit feedback 있다고 간주
    - X: (n_users, factors) -> 업데이트 대상
    - Y: (n_items, factors) -> 고정된 행렬
    - regularization: float
    """
    n_users, n_factors = X.shape
    # (factors x factors) 행렬
    YtY = Y.T @ Y
    initA = YtY + regularization * np.eye(n_factors, dtype=Y.dtype)
    initB = np.zeros(n_factors, dtype=Y.dtype)

    # CSR 인덱스
    indptr = Cui.indptr
    indices = Cui.indices
    data = Cui.data
    
    for u in range(n_users):
        start = indptr[u]
        end = indptr[u+1]
        length = end - start
        
        if length == 0:
            # 유저 u가 아무것도 클릭안했다면 0으로 초기화
            X[u, :] = 0
            continue
        
        # A, b 초기화
        A = initA.copy()
        b = initB.copy()

        # 해당 유저의 nonzero 아이템들에 대해 A, b 업데이트
        for idx in range(start, end):
            i = indices[idx]
            confidence = data[idx]

            if confidence > 0:
                # b += confidence * Y[i]
                b += confidence * Y[i]
            else:
                # confidence가 음수면, -confidence 로 치환
                confidence = -confidence

            # A += (confidence - 1) * (Yi^T * Yi)
            # np.outer(Y[i], Y[i]) = (factors x factors)
            A += (confidence - 1) * np.outer(Y[i], Y[i])

        # 선형시스템 A*x = b 해
        x = np.linalg.solve(A, b)

        X[u, :] = x

##


# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

class CustomImplicitALS:
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
        
        # 로거 설정
        self.logger = setup_logger('als')
    
    def _calculate_preference(self, row: pd.Series) -> float:
        """상호작용 타입에 따른 선호도 점수 계산"""
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
    
    def _prepare_matrices(self, interactions_df: pd.DataFrame) -> sp.csr_matrix:
        """상호작용 데이터를 선호도 행렬로 변환"""
        # 1. 유니크 사용자/상품 인덱싱
        unique_users = sorted(interactions_df["member_id"].unique())
        unique_items = sorted(interactions_df["product_id"].unique())
        
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = unique_users
        self.item2idx = {m: i for i, m in enumerate(unique_items)}
        self.idx2item = unique_items
        
        self.logger.info(f"사용자 수: {len(unique_users)}, 아이템 수: {len(unique_items)}")
        
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
        
        # 행렬 정보 로깅
        self.logger.info(f"\n행렬 변환 결과:")
        self.logger.info(f"- 행렬 크기: {P.shape}")
        self.logger.info(f"- 비영요소 수: {P.nnz}")
        self.logger.info(f"- 밀도: {P.nnz / (P.shape[0] * P.shape[1]):.4%}")
        
        # 데이터 타입 변환
        P.data = P.data.astype(np.float32)
        P.indices = P.indices.astype(np.int32)
        P.indptr = P.indptr.astype(np.int32)
        
        return P
    
    def train(self, interactions_df: pd.DataFrame) -> None:
        """모델 학습"""
        try:
            self.logger.info("ALS 모델 학습 시작")
            
            # 1. 선호도 행렬 준비
            P = self._prepare_matrices(interactions_df)
            Pt = P.T.tocsr()  # 전치행렬
            
            # 2. 잠재 요인 행렬 초기화
            np.random.seed(self.random_state)
            X = np.random.rand(len(self.user2idx), self.rank).astype(np.float32) * 0.01
            Y = np.random.rand(len(self.item2idx), self.rank).astype(np.float32) * 0.01
            
            # 3. ALS 반복 학습
            for iteration in range(self.max_iter):
                
                YtY = Y.T @ Y
                start = time.time()
                least_squares_python(
                    P,
                    X,   # 업데이트할 행렬
                    Y,   # 고정된 행렬
                    self.reg_param
                )
                least_squares_python(
                    Pt,
                    Y,   # 업데이트할 행렬
                    X,   # 고정된 행렬
                    self.reg_param
                )
                end = time.time()
                print(f"직접구현 걸린 시간: {end - start:.4f}초")
                
                # start = time.time()
                # _als.least_squares(
                #     P,
                #     X,   # 업데이트할 행렬
                #     Y,   # 고정된 행렬
                #     self.reg_param,
                #     num_threads=0
                # )
                # # (2) 사용자 행렬 갱신
                # XtX = X.T @ X
                # _als.least_squares(
                #     Pt,
                #     Y,   # 업데이트할 행렬
                #     X,   # 고정된 행렬
                #     self.reg_param,
                #     num_threads=0
                # )
                # end = time.time()
                # print(f"라이브러리 걸린 시간: {end - start:.4f}초")


                # 현재 RMSE 계산
                pred = X @ Y.T
                mask = P.toarray() > 0
                rmse = np.sqrt(np.mean((P.toarray()[mask] - pred[mask]) ** 2))
                self.logger.info(f"Iteration {iteration + 1}/{self.max_iter}, RMSE: {rmse:.4f}")
            
            self.user_factors = X
            self.item_factors = Y
            self.logger.info("모델 학습 완료")
            
            # 학습 결과 분석
            self.logger.info("\n=== 학습 결과 분석 ===")
            predictions = []
            
            # 실제 상호작용이 있는 데이터에 대해서만 분석
            for _, row in interactions_df.iterrows():
                user_idx = self.user2idx[row['member_id']]
                item_idx = self.item2idx[row['product_id']]
                actual_weight = self._calculate_preference(row)
                pred_score = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                
                predictions.append({
                    'user_id': row['member_id'],
                    'item_id': row['product_id'],
                    'interaction_type': row['interaction_type'],
                    'view_type': row['view_type'] if row['interaction_type'] == 'view' else None,
                    'actual_weight': actual_weight,
                    'predicted_score': pred_score
                })
            
            # 결과를 DataFrame으로 변환하고 정렬
            result_df = pd.DataFrame(predictions)
            result_df = result_df.sort_values('actual_weight', ascending=False)
            
            # CSV 파일로 저장
            output_path = 'output/weight_analysis.csv'
            os.makedirs('output', exist_ok=True)
            result_df.to_csv(output_path, index=False)
            
            # 상위 100개 결과 로깅
            self.logger.info("\n=== 상위 100개 가중치 비교 ===")
            self.logger.info("User ID | Item ID | Type | View Type | Actual Weight | Predicted Score")
            self.logger.info("-" * 75)
            
            for _, row in result_df.head(100).iterrows():
                view_type_str = f"type_{int(row['view_type'])}" if pd.notna(row['view_type']) else "N/A"
                view_type_display = view_type_str if row['interaction_type'] == 'view' else "-"
                
                self.logger.info(
                    f"{row['user_id']:7d} | "
                    f"{row['item_id']:7d} | "
                    f"{row['interaction_type']:<6s} | "
                    f"{view_type_display:^9s} | "
                    f"{row['actual_weight']:13.2f} | "
                    f"{row['predicted_score']:14.2f}"
                )
            
            self.logger.info(f"\n전체 분석 결과가 {output_path}에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"모델 학습 오류: {str(e)}")
            raise
    
    def generate_recommendations(self, top_n: int = 300) -> pd.DataFrame:
        """전체 사용자에 대한 추천 생성"""
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
            self.logger.error(f"추천 생성 오류: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.user_factors = None
        self.item_factors = None
        self.user2idx = {}
        self.idx2user = []
        self.item2idx = {}
        self.idx2item = [] 