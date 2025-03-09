"""
TF-IDF 기반 콘텐츠 필터링 모델

아이템의 텍스트 특징을 TF-IDF로 변환하여 
콘텐츠 기반 추천을 수행합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.base.base_recommender import BaseRecommender
import joblib

class TFIDFModel(BaseRecommender):
    """TF-IDF 기반 콘텐츠 필터링 모델"""
    
    def __init__(self, text_column: str = 'description'):
        """
        Args:
            text_column (str): 텍스트 특징이 있는 컬럼 이름
        """
        super().__init__()
        self.text_column = text_column
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        
    def train(self, items: pd.DataFrame) -> None:
        """모델 학습
        
        Args:
            items (pd.DataFrame): 아이템 특징 데이터
        """
        self.items = items
        self.item_ids = items.index.values
        
        # TF-IDF 특징 추출
        text_features = items[self.text_column].fillna('')
        self.tfidf_matrix = self.vectorizer.fit_transform(text_features)
        
        self.is_trained = True
        
    def get_recommendations(self, item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """특정 아이템과 유사한 아이템 추천
        
        Args:
            item_id (int): 아이템 ID
            k (int): 추천할 아이템 개수
            
        Returns:
            List[Tuple[int, float]]: (아이템 ID, 유사도 점수) 형태의 추천 리스트
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
            
        # 아이템 인덱스 찾기
        item_idx = np.where(self.item_ids == item_id)[0][0]
        
        # 유사도 계산
        item_vector = self.tfidf_matrix[item_idx]
        similarities = cosine_similarity(item_vector, self.tfidf_matrix)[0]
        
        # 유사도 기준 상위 k개 아이템 선택 (자기 자신 제외)
        similar_indices = similarities.argsort()[::-1][1:k+1]
        similar_items = [
            (int(self.item_ids[idx]), float(similarities[idx]))
            for idx in similar_indices
        ]
        
        return similar_items
    
    def save_model(self, path: str) -> None:
        """모델 저장"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'items': self.items,
            'item_ids': self.item_ids,
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path: str) -> None:
        """모델 로드"""
        checkpoint = joblib.load(path)
        self.vectorizer = checkpoint['vectorizer']
        self.tfidf_matrix = checkpoint['tfidf_matrix']
        self.items = checkpoint['items']
        self.item_ids = checkpoint['item_ids']
        self.is_trained = checkpoint['is_trained'] 