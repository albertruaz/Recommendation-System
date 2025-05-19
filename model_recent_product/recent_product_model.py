"""
최근 장바구니 상품 기반 추천 모델

사용자가 최근 장바구니에 넣은 상품들의 벡터를 기반으로 유사 상품을 추천합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from database.recommendation_db import RecommendationDB
from database.vector_db import VectorDB
import logging

class RecentProductModel:
    def __init__(self, top_n: int = 100):
        """
        최근 장바구니 상품 기반 추천 모델 초기화
        
        Args:
            top_n: 사용자별 추천할 상품 수
        """
        self.top_n = top_n
        self.vector_db = VectorDB()
        self.db = RecommendationDB()
        self.logger = logging.getLogger('recent_product_model')
        
    def get_user_cart_items(self, days: int = 30) -> pd.DataFrame:
        """
        사용자별 최근 장바구니 상품 목록 가져오기
        
        Args:
            days: 최근 몇 일 간의 데이터를 가져올지 설정
            
        Returns:
            사용자별 장바구니 상품 정보가 담긴 DataFrame
        """
        # DB에서 최근 장바구니 상품 정보 가져오기
        cart_items = self.db.get_recent_cart_items(days=days)
        return cart_items
    
    def get_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        상품 ID 목록에 대한 벡터 가져오기
        
        Args:
            product_ids: 상품 ID 목록
            
        Returns:
            상품 ID를 키로, 벡터를 값으로 하는 딕셔너리
        """
        # 벡터 DB에서 상품 벡터 가져오기
        product_vectors = self.vector_db.get_product_vectors(product_ids)
        return product_vectors
    
    def compute_average_vector(self, product_vectors: Dict[int, np.ndarray]) -> np.ndarray:
        """
        상품 벡터들의 평균 벡터 계산
        
        Args:
            product_vectors: 상품 ID를 키로, 벡터를 값으로 하는 딕셔너리
            
        Returns:
            평균 벡터
        """
        if not product_vectors:
            return None
            
        # 모든 벡터를 합하고 개수로 나누어 평균 계산
        vectors = list(product_vectors.values())
        converted_vectors = []
        
        # 각 벡터를 숫자형 배열로 변환
        for v in vectors:
            try:
                # 문자열인 경우 처리 (PostgreSQL 벡터 문자열 형식 '[1,2,3,...]' 처리)
                if isinstance(v, str):
                    # 문자열 벡터를 정제 ('[', ']' 제거 후 쉼표로 분리)
                    if v.startswith('[') and v.endswith(']'):
                        v = v[1:-1]  # '[', ']' 제거
                    v_list = [float(x.strip()) for x in v.split(',')]
                    converted_vectors.append(np.array(v_list, dtype=np.float32))
                # NumPy 배열이거나 변환 가능한 객체인 경우
                elif hasattr(v, '__array__') or hasattr(v, '__iter__'):
                    converted_vectors.append(np.array(v, dtype=np.float32))
                else:
                    self.logger.warning(f"벡터 변환 불가: {type(v)}")
                    continue
            except Exception as e:
                self.logger.warning(f"벡터 변환 중 오류 발생: {str(e)}, 타입: {type(v)}")
                continue
        
        # 변환된 벡터가 없으면 None 반환
        if not converted_vectors:
            return None
        
        # 평균 계산
        average_vector = np.mean(converted_vectors, axis=0)
        
        # 정규화 (길이가 1이 되도록)
        norm = np.linalg.norm(average_vector)
        if norm > 0:
            average_vector = average_vector / norm
            
        return average_vector
    
    def find_similar_products(self, query_vector: np.ndarray, exclude_ids: List[int] = None) -> pd.DataFrame:
        """
        쿼리 벡터와 유사한 상품들 찾기
        
        Args:
            query_vector: 검색 기준 벡터
            exclude_ids: 결과에서 제외할 상품 ID 목록
            
        Returns:
            유사 상품 DataFrame (product_id, score 컬럼 포함)
        """
        if query_vector is None:
            return pd.DataFrame(columns=['product_id', 'score'])
            
        # Vector DB에서 유사 상품 검색
        similar_products = self.vector_db.search_similar_products(
            query_vector, 
            top_n=self.top_n,
            exclude_ids=exclude_ids or []
        )
        
        return similar_products
    
    def generate_recommendations(self, days: int = 30) -> pd.DataFrame:
        """
        모든 사용자에 대한 추천 생성
        
        Args:
            days: 최근 몇 일 간의 장바구니 데이터를 사용할지
            
        Returns:
            사용자별 추천 상품 목록이 담긴 DataFrame
        """
        # 1. 사용자별 최근 장바구니 상품 가져오기
        cart_items = self.get_user_cart_items(days=days)
        
        # 결과를 저장할 DataFrame
        recommendations = []
        
        # 사용자별로 그룹화하여 처리
        for member_id, user_items in cart_items.groupby('member_id'):
            # 사용자의 장바구니 상품 ID 목록
            product_ids = user_items['product_id'].unique().tolist()
            
            try:
                # 2. 장바구니 상품 벡터 가져오기
                product_vectors = self.get_product_vectors(product_ids)
                
                # 벡터가 없는 경우 다음 사용자로 넘어감
                if not product_vectors:
                    continue
                
                # 3. 평균 벡터
                avg_vector = self.compute_average_vector(product_vectors)
                
                # 4. 유사 상품 찾기 (본인이 담은 상품은 제외)
                similar_products = self.find_similar_products(avg_vector, exclude_ids=product_ids)
                
                # 추천 결과에 사용자 ID 추가
                if not similar_products.empty:
                    similar_products['member_id'] = member_id
                    recommendations.append(similar_products)
            except Exception as e:
                self.logger.error(f"사용자 {member_id}의 추천 생성 중 오류 발생: {str(e)}")
                continue
        
        # 모든 사용자의 추천 결과 합치기
        if recommendations:
            recommendations_df = pd.concat(recommendations, ignore_index=True)
            # 컬럼 순서 정리
            recommendations_df = recommendations_df[['member_id', 'product_id', 'score']]
            return recommendations_df
        else:
            return pd.DataFrame(columns=['member_id', 'product_id', 'score']) 