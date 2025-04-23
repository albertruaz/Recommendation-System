"""
ALS 추천 시스템의 기본 클래스

이 모듈은 ALS 알고리즘의 기본 기능을 제공하는 추상 클래스를 정의합니다.
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
import os
from abc import ABC, abstractmethod
from utils.logger import setup_logger

class BaseALS(ABC):
    """ALS 기반 추천 시스템의 기본 클래스"""
    
    def __init__(self, max_iter: int = 15, reg_param: float = 0.1,
                 rank: int = 10, random_state: int = 42, interaction_weights: dict = None):
        """
        Args:
            max_iter (int): 최대 반복 횟수
            reg_param (float): 정규화 파라미터
            rank (int): 잠재 요인 개수
            random_state (int): 랜덤 시드
            interaction_weights (dict): 상호작용 타입별 가중치
        """
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.random_state = random_state
        self.interaction_weights = interaction_weights or {
            "impression1": 0.0,
            "impression2": 0.0,
            "view1": 1.0,
            "view2": 2.0,
            "like": 5.0,
            "cart": 7.0,
            "purchase": 10.0,
            "review": 10.0
        }
        
        self.user_factors = None
        self.item_factors = None
        self.user2idx = {}
        self.idx2user = []
        self.item2idx = {}
        self.idx2item = []
        
        # 로거 설정
        self.logger = setup_logger('als')
    
    def _prepare_indices(self, interactions_df: pd.DataFrame) -> None:
        """사용자와 아이템 인덱스 매핑 생성"""
        unique_users = sorted(interactions_df["member_id"].unique())
        unique_items = sorted(interactions_df["product_id"].unique())
        
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = unique_users
        self.item2idx = {m: i for i, m in enumerate(unique_items)}
        self.idx2item = unique_items
        
        self.logger.info(f"사용자 수: {len(unique_users)}, 아이템 수: {len(unique_items)}")
    
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
    
    def prepare_matrices(self, interactions_df: pd.DataFrame):
        """상호작용 데이터를 행렬 형태로 변환
        
        이 메서드는 상호작용 데이터프레임을 입력받아 행렬 형태로 변환합니다.
        하위 클래스에서 이 메서드를 오버라이드하여 특정 구현에 맞게 변환할 수 있습니다.
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터프레임
            
        Returns:
            행렬 형태의 데이터(구현에 따라 sp.csr_matrix, MatrixMarket 등)
        """
        # 인덱스 매핑 생성
        self._prepare_indices(interactions_df)
        
        # 가중치 계산
        interactions_df["weight"] = interactions_df.apply(self._get_interaction_weight, axis=1)
        
        # 사용자-상품별로 가중치 합산
        aggregated_df = interactions_df.groupby(
            ["member_id", "product_id"]
        )["weight"].max().reset_index()
        
        # 인덱스 매핑 적용 및 행렬 생성
        user_indices = aggregated_df['member_id'].map(self.user2idx).values
        item_indices = aggregated_df['product_id'].map(self.item2idx).values
        weights = aggregated_df['weight'].values
        
        # Confidence 적용 (alpha 파라미터가 있는 경우)
        if hasattr(self, 'alpha'):
            confidence = self.alpha * weights
        else:
            confidence = weights
        
        # COO 형식으로 변환
        sparse_matrix = sp.coo_matrix(
            (confidence, (user_indices, item_indices)),
            shape=(len(self.user2idx), len(self.item2idx))
        ).tocsr()
        
        
        return sparse_matrix
    
    @abstractmethod
    def train(self, interactions_df, matrix_data) -> None:
        """모델 학습
        
        Args:
            interactions_df: 상호작용 데이터프레임
            matrix_data: 준비된 행렬 데이터
        """
        pass
    
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
            self.logger.info(f"추천 생성 완료: {len(recommendations_df)}개의 추천")
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
        self.logger.info("리소스 정리 완료")
    
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