"""
ALS 추천 시스템의 기본 클래스

이 모듈은 ALS 알고리즘의 기본 기능을 제공하는 추상 클래스를 정의합니다.
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from abc import ABC, abstractmethod
from utils.logger import setup_logger

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

class BaseALS(ABC):
    """ALS 기반 추천 시스템의 기본 클래스"""
    
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
    
    @abstractmethod
    def _prepare_matrices(self, interactions_df: pd.DataFrame) -> sp.csr_matrix:
        """상호작용 데이터를 행렬로 변환"""
        pass
    
    @abstractmethod
    def train(self, interactions_df: pd.DataFrame) -> None:
        """모델 학습"""
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