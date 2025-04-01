"""
Implicit ALS 추천 시스템 모델

이 모듈은 Implicit ALS를 사용하여 사용자-상품 추천을 생성하는 클래스를 제공합니다.
상호작용 가중치는 config/rating_weights.py에 정의되어 있습니다.
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from .base_als import BaseALS, CONFIG
import os

# numpy 출력 설정
np.set_printoptions(threshold=np.inf, precision=3, suppress=True)

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

class ImplicitALS(BaseALS):
    """Implicit ALS 기반 추천 시스템 클래스"""
    
    def __init__(self, max_iter: int = 15, reg_param: float = 0.1,
                 rank: int = 100, alpha: float = 40):
        super().__init__(max_iter, reg_param, rank)
        self.alpha = alpha
        self.model = None
    
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
    
    def _prepare_matrices(self, interactions_df: pd.DataFrame) -> sp.csr_matrix:
        """상호작용 데이터를 sparse matrix로 변환"""
        self._prepare_indices(interactions_df)
        
        # 가중치 계산
        interactions_df["weight"] = interactions_df.apply(self._get_interaction_weight, axis=1)
        
        # 사용자-상품별로 가중치 합산
        aggregated_df = interactions_df.groupby(
            ["member_id", "product_id"]
        )["weight"].max().reset_index()
        
        # Confidence 적용 및 sparse matrix 생성
        row = aggregated_df["member_id"].map(self.user2idx)
        col = aggregated_df["product_id"].map(self.item2idx)
        data = aggregated_df["weight"].apply(lambda x: 1.0 + self.alpha * x)
        
        matrix = sp.coo_matrix(
            (data, (row, col)),
            shape=(len(self.user2idx), len(self.item2idx))
        ).tocsr()
        
        # 행렬 정보 로깅
        self.logger.info(f"\n행렬 변환 결과:")
        self.logger.info(f"- 행렬 크기: {matrix.shape}")
        self.logger.info(f"- 비영요소 수: {matrix.nnz}")
        self.logger.info(f"- 밀도: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")
        
        # 샘플 데이터 출력 (처음 10x10 행렬)
        sample_size = 10
        sample_matrix = matrix[:sample_size, :sample_size].toarray()
        
        # 행과 열 인덱스 준비
        row_indices = [f"User {self.idx2user[i]}" for i in range(sample_size)]
        col_indices = [f"Item {self.idx2item[i]}" for i in range(sample_size)]
        
        # 행렬 출력 준비
        matrix_str = "\n행렬 샘플 (10x10):\n"
        
        # 열 헤더 추가
        max_user_len = max(len(str(u)) for u in row_indices)
        matrix_str += " " * (max_user_len + 2)  # 왼쪽 여백
        for col in col_indices:
            matrix_str += f"{col:>10} "  # 각 열에 10자리 할당
        matrix_str += "\n"
        
        # 데이터 행 추가
        for i, user in enumerate(row_indices):
            matrix_str += f"{user:<{max_user_len}} |"  # 사용자 ID와 구분선
            for j in range(sample_size):
                matrix_str += f"{sample_matrix[i,j]:10.3f} "  # 각 값에 10자리 할당
            matrix_str += "\n"
        
        self.logger.info(matrix_str)
        
        return matrix
    
    def train(self, interactions_df: pd.DataFrame) -> float:
        """모델 학습"""
        try:
            self.logger.info("Implicit ALS 모델 학습 시작")
            
            # 데이터 준비
            self.train_matrix = self._prepare_matrices(interactions_df)
            
            # 모델 초기화 및 학습
            self.model = AlternatingLeastSquares(
                factors=self.rank,
                regularization=self.reg_param,
                iterations=self.max_iter
            )
            self.model.fit(self.train_matrix)
            
            # 학습된 잠재 요인 저장
            self.user_factors = self.model.user_factors
            self.item_factors = self.model.item_factors
            
            self.logger.info("모델 학습 완료")

            # 학습 결과 분석
            self.logger.info("\n=== 학습 결과 분석 ===")
            
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
                    'view_type': row['view_type'] if row['interaction_type'] == 'view' else None,
                    'actual_weight': row['actual_weight'],
                    'predicted_score': pred_score
                })
            
            # 결과를 DataFrame으로 변환하고 정렬
            result_df = pd.DataFrame(predictions)
            result_df = result_df.sort_values('actual_weight', ascending=False)
            
            # 출력 디렉토리 생성
            os.makedirs('output', exist_ok=True)
            
            # CSV 파일로 저장
            output_path = 'output/weight_analysis.csv'
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
            
            # 상호작용 타입별 통계
            self.logger.info("\n=== 상호작용 타입별 통계 ===")
            type_stats = result_df.groupby('interaction_type').agg({
                'actual_weight': ['mean', 'min', 'max', 'count'],
                'predicted_score': ['mean', 'min', 'max']
            }).round(3)
            
            self.logger.info("\n" + str(type_stats))
            self.logger.info(f"\n전체 분석 결과가 {output_path}에 저장되었습니다.")
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"모델 학습 오류: {str(e)}")
            raise
    
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
        super().cleanup()
        self.model = None
        self.train_matrix = None 