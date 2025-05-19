"""
Vector Database 관리 모듈

상품 임베딩 벡터 및 벡터 검색 기능을 제공합니다.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from .vector_db_connector import VectorDBConnector

class VectorDB:
    def __init__(self):
        """
        Vector Database 초기화
        """
        # PostgreSQL 기반 Vector DB 연결
        self.connector = VectorDBConnector()
        
    def get_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        상품 ID 목록에 대한 벡터 가져오기
        
        Args:
            product_ids: 상품 ID 목록
            
        Returns:
            상품 ID를 키로, 벡터를 값으로 하는 딕셔너리
        """
        # PostgreSQL에서 벡터 데이터 가져오기
        return self.connector.get_product_vectors(product_ids)
    
    def search_similar_products(self, query_vector: np.ndarray, top_n: int = 100, exclude_ids: List[int] = None) -> pd.DataFrame:
        """
        쿼리 벡터와 유사한 상품 검색
        
        Args:
            query_vector: 검색 기준 벡터
            top_n: 반환할 최대 결과 수
            exclude_ids: 결과에서 제외할 상품 ID 목록
            
        Returns:
            유사 상품 DataFrame (product_id, score 컬럼 포함)
        """
        try:
            # PostgreSQL에서 유사 상품 검색
            similar_products = self.connector.search_by_vector(
                query_vector=query_vector, 
                top_k=top_n,
                exclude_ids=exclude_ids
            )
            
            # 결과를 DataFrame으로 변환
            if similar_products:
                result_df = pd.DataFrame({
                    'product_id': [p[0] for p in similar_products],
                    'score': [1.0 - p[1] for p in similar_products]  # 거리를 유사도로 변환 (1 - 거리)
                })
                return result_df
            else:
                return pd.DataFrame(columns=['product_id', 'score'])
                
        except Exception as e:
            print(f"유사 상품 검색 중 오류 발생: {str(e)}")
            return pd.DataFrame(columns=['product_id', 'score'])
    
    def get_similar_products_by_id(self, product_id: int, top_n: int = 100) -> pd.DataFrame:
        """
        특정 상품 ID와 유사한 상품 검색
        
        Args:
            product_id: 상품 ID
            top_n: 반환할 최대 결과 수
            
        Returns:
            유사 상품 DataFrame (product_id, score 컬럼 포함)
        """
        try:
            # PostgreSQL에서 유사 상품 검색
            similar_products = self.connector.get_similar_products_by_id(
                product_id=product_id,
                top_k=top_n
            )
            
            # 결과를 DataFrame으로 변환
            if similar_products:
                result_df = pd.DataFrame({
                    'product_id': [p[0] for p in similar_products],
                    'score': [1.0 - p[1] for p in similar_products]  # 거리를 유사도로 변환 (1 - 거리)
                })
                return result_df
            else:
                return pd.DataFrame(columns=['product_id', 'score'])
                
        except Exception as e:
            print(f"유사 상품 검색 중 오류 발생: {str(e)}")
            return pd.DataFrame(columns=['product_id', 'score']) 