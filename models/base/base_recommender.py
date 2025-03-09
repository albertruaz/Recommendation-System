"""
추천 시스템의 기본 인터페이스

모든 추천 모델은 이 기본 클래스를 상속받아 구현해야 합니다.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class BaseRecommender(ABC):
    """추천 시스템 기본 클래스"""
    
    def __init__(self):
        """모델 초기화"""
        self.is_trained = False
    
    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """모델 학습"""
        pass
    
    @abstractmethod
    def get_recommendations(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """특정 사용자에 대한 추천 아이템 생성
        
        Args:
            user_id (int): 사용자 ID
            k (int): 추천할 아이템 개수
            
        Returns:
            List[Tuple[int, float]]: (아이템 ID, 점수) 형태의 추천 리스트
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """모델 저장"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """모델 로드"""
        pass 