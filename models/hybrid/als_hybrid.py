"""
ALS 기반 하이브리드 추천 모델

ALS 모델과 콘텐츠 기반 추천의 결과를 결합하는 하이브리드 모델입니다.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict
from models.base.base_recommender import BaseRecommender
from models.content.tfidf import TFIDFModel
from utils.als_connector import get_relationship_weights

class ALSHybridModel(BaseRecommender):
    """ALS 기반 하이브리드 추천 모델"""
    
    def __init__(self, als_weight: float = 0.7, config_dict=None):
        """
        Args:
            als_weight (float): ALS 모델의 가중치 (0~1)
            config_dict (dict, optional): 모델 설정
        """
        super().__init__()
        self.config = config_dict or {}
        self.als_weight = als_weight
        self.content_weight = 1 - als_weight
        
        # 콘텐츠 기반 모델 초기화
        self.content_model = TFIDFModel()
        self.is_trained = False
        
    def train(self, ratings: pd.DataFrame, items: pd.DataFrame,
              validation_data: pd.DataFrame = None) -> None:
        """모델 학습
        
        Args:
            ratings (pd.DataFrame): 평점 데이터
            items (pd.DataFrame): 아이템 특징 데이터
            validation_data (pd.DataFrame, optional): 검증 데이터
        """
        # 콘텐츠 기반 모델 학습
        logging.info("콘텐츠 기반 모델 학습 중...")
        self.content_model.train(items)
        
        # ALS 모델은 utils.als_connector를 통해 접근하므로 여기서 학습하지 않음
        
        if validation_data is not None:
            self._optimize_weights(validation_data)
            
        self.is_trained = True
        logging.info("하이브리드 모델 학습 완료")
        
    def _optimize_weights(self, validation_data: pd.DataFrame):
        """검증 데이터를 사용해 최적의 가중치 탐색
        
        Args:
            validation_data (pd.DataFrame): 검증 데이터
        """
        best_ndcg = 0
        best_weight = self.als_weight
        
        for weight in np.arange(0.1, 1.0, 0.1):
            self.als_weight = weight
            self.content_weight = 1 - weight
            
            # 검증 데이터로 성능 평가
            ndcg_scores = []
            for user_id in validation_data['user_id'].unique()[:20]:  # 일부만 사용
                true_items = validation_data[
                    validation_data['user_id'] == user_id
                ]['item_id'].tolist()
                
                recs = self.get_recommendations(user_id, k=10)
                rec_items = [item_id for item_id, _ in recs]
                
                # NDCG 계산
                relevance = [1 if item in true_items else 0 for item in rec_items]
                ndcg = self._calculate_ndcg(relevance)
                ndcg_scores.append(ndcg)
            
            avg_ndcg = np.mean(ndcg_scores)
            if avg_ndcg > best_ndcg:
                best_ndcg = avg_ndcg
                best_weight = weight
        
        self.als_weight = best_weight
        self.content_weight = 1 - best_weight
        logging.info(f"최적 가중치 - ALS: {self.als_weight:.2f}, 콘텐츠: {self.content_weight:.2f}")
    
    def _calculate_ndcg(self, relevance: List[int], k: int = 10) -> float:
        """NDCG 계산
        
        Args:
            relevance (List[int]): 관련성 점수 리스트
            k (int): 추천 아이템 수
            
        Returns:
            float: NDCG 점수
        """
        dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance[:k]))
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True)[:k]))
        return dcg / idcg if idcg > 0 else 0
    
    def get_recommendations(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """특정 사용자에 대한 추천 생성
        
        Args:
            user_id (int): 사용자 ID
            k (int): 추천할 아이템 개수
            
        Returns:
            List[Tuple[int, float]]: (아이템 ID, 점수) 형태의 추천 리스트
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        # 콘텐츠 기반 추천 (최근 상호작용한 아이템 기반)
        content_recs = self.content_model.get_recommendations(user_id, k=k*2)
        content_items = [item_id for item_id, _ in content_recs]
        
        # ALS 관계 가중치 계산
        als_weights = get_relationship_weights(user_id, content_items)
        
        # 점수 결합
        final_scores = {}
        for item_id, content_score in content_recs:
            als_score = als_weights.get(item_id, 0)
            final_scores[item_id] = (
                self.als_weight * als_score + 
                self.content_weight * content_score
            )
        
        # 상위 k개 추천
        return sorted(
            [(item_id, score) for item_id, score in final_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]
    
    def save_model(self, path: str) -> None:
        """모델 저장"""
        import os
        import joblib
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 콘텐츠 모델 저장
        self.content_model.save_model(f"{path}_content")
        
        # 가중치 저장
        save_dict = {
            'als_weight': self.als_weight,
            'content_weight': self.content_weight,
            'is_trained': self.is_trained
        }
        joblib.dump(save_dict, f"{path}_weights")
        
        logging.info(f"하이브리드 모델 저장 완료: {path}")
        
    def load_model(self, path: str) -> None:
        """모델 로드"""
        import joblib
        
        # 콘텐츠 모델 로드
        self.content_model.load_model(f"{path}_content")
        
        # 가중치 로드
        save_dict = joblib.load(f"{path}_weights")
        self.als_weight = save_dict['als_weight']
        self.content_weight = save_dict['content_weight']
        self.is_trained = save_dict['is_trained']
        
        logging.info(f"하이브리드 모델 로드 완료: {path}") 