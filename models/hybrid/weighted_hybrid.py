"""
가중치 기반 하이브리드 추천 모델

협업 필터링(BPR)과 콘텐츠 기반(TF-IDF) 추천의 
결과를 가중 결합하는 하이브리드 모델입니다.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from models.base.base_recommender import BaseRecommender
from models.collaborative.bpr import BPRModel
from models.content.tfidf import TFIDFModel

class WeightedHybridModel(BaseRecommender):
    """가중치 기반 하이브리드 추천 모델"""
    
    def __init__(self, cf_weight: float = 0.7):
        """
        Args:
            cf_weight (float): 협업 필터링 모델의 가중치 (0~1)
        """
        super().__init__()
        self.cf_weight = cf_weight
        self.content_weight = 1 - cf_weight
        
        self.cf_model = BPRModel()
        self.content_model = TFIDFModel()
        
    def train(self, ratings: pd.DataFrame, items: pd.DataFrame,
              validation_data: pd.DataFrame = None) -> None:
        """모델 학습
        
        Args:
            ratings (pd.DataFrame): 평점 데이터
            items (pd.DataFrame): 아이템 특징 데이터
            validation_data (pd.DataFrame, optional): 검증 데이터
        """
        # 협업 필터링 모델 학습
        self.cf_model.train(ratings)
        
        # 콘텐츠 기반 모델 학습
        self.content_model.train(items)
        
        if validation_data is not None:
            self._optimize_weights(validation_data)
            
        self.is_trained = True
        
    def _optimize_weights(self, validation_data: pd.DataFrame):
        """검증 데이터를 사용해 최적의 가중치 탐색
        
        Args:
            validation_data (pd.DataFrame): 검증 데이터
        """
        best_ndcg = 0
        best_weight = self.cf_weight
        
        for weight in np.arange(0.1, 1.0, 0.1):
            self.cf_weight = weight
            self.content_weight = 1 - weight
            
            # 검증 데이터로 성능 평가
            ndcg_scores = []
            for user_id in validation_data['user_id'].unique():
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
        
        self.cf_weight = best_weight
        self.content_weight = 1 - best_weight
    
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
        
        # 협업 필터링 추천
        cf_recs = self.cf_model.get_recommendations(user_id, k=k)
        cf_scores = {item_id: score for item_id, score in cf_recs}
        
        # 콘텐츠 기반 추천 (최근 상호작용한 아이템 기반)
        content_scores = {}
        for item_id in cf_scores.keys():
            content_recs = self.content_model.get_recommendations(item_id, k=k)
            for rec_item_id, score in content_recs:
                if rec_item_id not in content_scores:
                    content_scores[rec_item_id] = score
                else:
                    content_scores[rec_item_id] = max(content_scores[rec_item_id], score)
        
        # 점수 결합
        final_scores = {}
        all_items = set(cf_scores.keys()) | set(content_scores.keys())
        
        for item_id in all_items:
            cf_score = cf_scores.get(item_id, 0)
            content_score = content_scores.get(item_id, 0)
            final_scores[item_id] = (
                self.cf_weight * cf_score + 
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
        self.cf_model.save_model(f"{path}_cf")
        self.content_model.save_model(f"{path}_content")
        np.save(f"{path}_weights", [self.cf_weight, self.content_weight])
        
    def load_model(self, path: str) -> None:
        """모델 로드"""
        self.cf_model.load_model(f"{path}_cf")
        self.content_model.load_model(f"{path}_content")
        weights = np.load(f"{path}_weights.npy")
        self.cf_weight = float(weights[0])
        self.content_weight = float(weights[1])
        self.is_trained = True 