"""
Implicit ALS 추천 시스템 모델

이 모듈은 Implicit ALS를 사용하여 사용자-상품 추천을 생성하는 클래스를 제공합니다.
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from .base_als import BaseALS
import os

# numpy 출력 설정
np.set_printoptions(threshold=np.inf, precision=3, suppress=True)

class ImplicitALS(BaseALS):
    """Implicit ALS 기반 추천 시스템 클래스"""
    
    def __init__(self, max_iter: int = 15, reg_param: float = 0.1,
                 rank: int = 100, random_state: int = 42, alpha: float = 40,
                 interaction_weights: dict = None):
        super().__init__(max_iter, reg_param, rank, random_state, interaction_weights)
        self.alpha = alpha
        self.model = None
        self.train_matrix = None
    
    def train(self, interactions_df, matrix_data) -> float:
        """모델 학습
        
        Args:
            interactions_df: 상호작용 데이터프레임
            matrix_data: 준비된 학습 행렬 (sparse matrix)
        """
        try:
            self.logger.info("Implicit ALS 모델 학습 시작")
            
            # 데이터 설정 - 항상 준비된 행렬 데이터 사용
            self.train_matrix = matrix_data
            
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

            # 학습 결과 분석 및 로깅
            self._log_training_results(interactions_df)
            
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