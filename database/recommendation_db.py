"""
추천 시스템과 데이터베이스 연결을 위한 유틸리티 모듈

이 모듈은 데이터베이스에서 사용자-아이템 상호작용 데이터를 가져오고,
추천 모델에 필요한 형식으로 변환하는 기능을 제공합니다.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
from .db_connector import DBConnector
from .vector_db_connector import VectorDBConnector
from sqlalchemy import text
from utils.config import load_config

class RecommendationDB:
    """추천 시스템을 위한 데이터베이스 유틸리티 클래스"""
    
    def __init__(self):
        """데이터베이스 연결 초기화"""
        self.db = DBConnector()
        self.vector_db = VectorDBConnector()
        self.logger = logging.getLogger(__name__)
        self.config = load_config()
    
    def _build_cte_queries(self, start_date: str) -> Dict[str, str]:
        """각 상호작용 타입별 CTE 쿼리를 생성합니다."""
        return {
            'purchases': f"""
                purchases AS (
                    SELECT member_id, product_id, COUNT(*) as purchase_count
                    FROM orders o
                    JOIN order_items oi ON o.id = oi.order_id
                    WHERE o.created_at >= '{start_date}'
                    AND o.status = 'completed'
                    GROUP BY member_id, product_id
                )
            """,
            'cart_items': f"""
                cart_items AS (
                    SELECT member_id, product_id, COUNT(*) as cart_count
                    FROM cart_items
                    WHERE created_at >= '{start_date}'
                    GROUP BY member_id, product_id
                )
            """,
            'likes': f"""
                likes AS (
                    SELECT member_id, product_id, COUNT(*) as like_count
                    FROM likes
                    WHERE created_at >= '{start_date}'
                    GROUP BY member_id, product_id
                )
            """,
            'views': f"""
                views AS (
                    SELECT member_id, product_id, COUNT(*) as view_count
                    FROM product_views
                    WHERE created_at >= '{start_date}'
                    GROUP BY member_id, product_id
                )
            """,
            'impressions': f"""
                impressions AS (
                    SELECT member_id, product_id, COUNT(*) as impression_count
                    FROM product_impressions
                    WHERE created_at >= '{start_date}'
                    GROUP BY member_id, product_id
                )
            """
        }
    
    def _build_where_conditions(self) -> str:
        """WHERE 절 조건을 생성합니다."""
        return """
            WHERE (
                p.purchase_count > 0 OR
                c.cart_count > 0 OR
                l.like_count > 0 OR
                v.view_count >= 3 OR
                i.impression_count >= 5
            )
        """
    
    def get_user_item_interactions(self, days: Optional[int] = None) -> pd.DataFrame:
        """
        사용자-상품 상호작용 데이터를 가져옵니다.
        
        Args:
            days (Optional[int]): 최근 몇 일간의 데이터를 가져올지 지정
            
        Returns:
            pd.DataFrame: 상호작용 데이터 (member_id, product_id, rating)
        """
        try:
            # 시작 날짜 계산
            date_condition = f"CURRENT_DATE - INTERVAL '{days} days'" if days else "CURRENT_DATE - INTERVAL '30 days'"
            
            # CTE 쿼리 생성
            cte_queries = self._build_cte_queries(date_condition)
            
            # 메인 쿼리 생성
            query = f"""
                WITH {','.join(cte_queries.values())}
                SELECT 
                    COALESCE(p.member_id, c.member_id, l.member_id, v.member_id, i.member_id) as member_id,
                    COALESCE(p.product_id, c.product_id, l.product_id, v.product_id, i.product_id) as product_id,
                    GREATEST(
                        COALESCE(CASE WHEN p.purchase_count > 0 THEN 5.0 END, 0),
                        COALESCE(CASE WHEN c.cart_count > 0 THEN 4.0 END, 0),
                        COALESCE(CASE WHEN l.like_count > 0 THEN 3.0 END, 0),
                        COALESCE(CASE WHEN v.view_count >= 3 THEN 2.0 END, 0),
                        COALESCE(CASE WHEN i.impression_count >= 5 THEN 1.0 END, 0)
                    ) as rating
                FROM purchases p
                FULL OUTER JOIN cart_items c USING (member_id, product_id)
                FULL OUTER JOIN likes l USING (member_id, product_id)
                FULL OUTER JOIN views v USING (member_id, product_id)
                FULL OUTER JOIN impressions i USING (member_id, product_id)
            """
            
            # WHERE 조건 추가
            query += self._build_where_conditions()
            
            # 쿼리 실행
            with self.db.get_connection() as conn:
                interactions = pd.read_sql(query, conn)
            
            self.logger.info(f"총 {len(interactions)}개의 상호작용 데이터를 가져왔습니다.")
            return interactions
            
        except Exception as e:
            self.logger.error(f"상호작용 데이터 가져오기 실패: {str(e)}")
            raise
    
    def get_item_features(self) -> pd.DataFrame:
        """
        상품 특성 데이터를 가져옵니다.
        
        Returns:
            pd.DataFrame: 상품 특성 데이터
        """
        try:
            query = """
                SELECT 
                    id as product_id,
                    title,
                    description,
                    primary_category_id,
                    secondary_category_id,
                    price,
                    status
                FROM product
                WHERE status = 'SALE'
            """
            
            with self.db.get_connection() as conn:
                df = pd.read_sql(text(query), conn)
                self.logger.info(f"총 {len(df)}개의 상품 특성 데이터 로드 완료")
                return df
                
        except Exception as e:
            self.logger.error(f"상품 특성 데이터 가져오기 실패: {str(e)}")
            raise
    
    def get_item_embeddings(self, item_ids: Optional[List[int]] = None) -> Dict[int, List[float]]:
        """
        아이템 임베딩 벡터를 가져옵니다.
        
        Args:
            item_ids (List[int], optional): 가져올 아이템 ID 목록. None이면 모든 아이템 가져옴.
            
        Returns:
            Dict[int, List[float]]: {아이템 ID: 임베딩 벡터} 형태의 딕셔너리
        """
        self.logger.info("아이템 임베딩 벡터 가져오기")
        
        session = self.vector_db.Session()
        try:
            if item_ids:
                query = """
                    SELECT id, image_vector
                    FROM product
                    WHERE id = ANY(:item_ids)
                """
                params = {'item_ids': item_ids}
            else:
                query = """
                    SELECT id, image_vector
                    FROM product
                """
                params = {}
            
            result = session.execute(text(query), params)
            embeddings = {
                row.id: self._parse_vector(row.image_vector)
                for row in result
            }
            
            self.logger.info(f"총 {len(embeddings)}개의 아이템 임베딩 로드 완료")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"임베딩 데이터 가져오기 실패: {str(e)}")
            raise
        finally:
            session.close()
    
    def _parse_vector(self, vector_str: str) -> List[float]:
        if isinstance(vector_str, str):
            return [float(x) for x in vector_str.strip('[]').split(',')]
        return vector_str

    def prepare_training_data(self, days: int = 90, test_ratio: float = 0.2, 
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        추천 모델 학습을 위한 데이터를 준비합니다.
        
        Args:
            days (int): 최근 몇 일 동안의 데이터를 가져올지 지정
            test_ratio (float): 테스트 데이터 비율
            random_state (int): 랜덤 시드
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (학습 데이터, 테스트 데이터, 아이템 특성 데이터)
        """
        # 1. 사용자-아이템 상호작용 데이터 가져오기
        interactions = self.get_user_item_interactions(days=days)
        
        # 2. 아이템 특성 데이터 가져오기
        items = self.get_item_features()
        
        # 3. 데이터 전처리
        # 3.1. 중복 제거 (같은 사용자-아이템 쌍이 여러 번 있을 경우 최대 rating 사용)
        interactions = interactions.sort_values('rating', ascending=False)
        interactions = interactions.drop_duplicates(subset=['member_id', 'product_id'])
        
        # 3.2. 아이템 필터링 (특성 데이터에 있는 아이템만 사용)
        valid_items = set(items['product_id'])
        interactions = interactions[interactions['product_id'].isin(valid_items)]
        
        # 3.3. ID 매핑 (연속적인 정수 ID로 변환)
        user_ids = interactions['member_id'].unique()
        item_ids = interactions['product_id'].unique()
        
        user_id_map = {uid: i for i, uid in enumerate(user_ids)}
        item_id_map = {iid: i for i, iid in enumerate(item_ids)}
        
        interactions['member_id_mapped'] = interactions['member_id'].map(user_id_map)
        interactions['product_id_mapped'] = interactions['product_id'].map(item_id_map)
        items['product_id_mapped'] = items['product_id'].map(item_id_map)
        
        # 3.4. 원본 ID를 별도 컬럼으로 저장
        interactions['member_id_original'] = interactions['member_id']
        interactions['product_id_original'] = interactions['product_id']
        items['product_id_original'] = items['product_id']
        
        # 3.5. 매핑된 ID를 기본 ID로 사용
        interactions['member_id'] = interactions['member_id_mapped']
        interactions['product_id'] = interactions['product_id_mapped']
        items['product_id'] = items['product_id_mapped']
        
        # 4. 학습/테스트 분할
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(
            interactions,
            test_size=test_ratio,
            random_state=random_state,
            stratify=interactions['member_id']  # 사용자별로 균등하게 분할
        )
        
        # 5. 필요한 컬럼만 선택
        train = train[['member_id', 'product_id', 'rating', 'member_id_original', 'product_id_original']]
        test = test[['member_id', 'product_id', 'rating', 'member_id_original', 'product_id_original']]
        
        self.logger.info(f"학습 데이터: {len(train)}개, 테스트 데이터: {len(test)}개")
        self.logger.info(f"고유 사용자 수: {train['member_id'].nunique()}")
        self.logger.info(f"고유 아이템 수: {train['product_id'].nunique()}")
        
        # 6. ID 매핑 정보 저장
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.reverse_user_id_map = {v: k for k, v in user_id_map.items()}
        self.reverse_item_id_map = {v: k for k, v in item_id_map.items()}
        
        return train, test, items
    
    def save_recommendations(self, recommendations_df: pd.DataFrame, model_name: str = "als") -> None:
        """
        추천 결과를 데이터베이스에 저장합니다.
        
        Args:
            recommendations_df (pd.DataFrame): 추천 결과 데이터프레임
            model_name (str): 모델 이름
        """
        try:
            with self.db.get_connection() as conn:
                # 기존 추천 결과 삭제
                conn.execute(
                    text("DELETE FROM model_recommendations WHERE model_name = :model_name"),
                    {'model_name': model_name}
                )
                
                # 새로운 추천 결과 저장
                recommendations_df['model_name'] = model_name
                recommendations_df.to_sql(
                    'model_recommendations',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                self.logger.info(f"총 {len(recommendations_df)}개의 추천 결과 저장 완료")
                
        except Exception as e:
            self.logger.error(f"추천 결과 저장 실패: {str(e)}")
            raise
    
    def get_recommendations_for_user(
        self, 
        user_id: int, 
        model_name: str = "als",
        limit: int = 10
    ) -> List[Tuple[int, float]]:
        """
        특정 사용자에 대한 추천 결과를 가져옵니다.
        
        Args:
            user_id (int): 사용자 ID
            model_name (str): 모델 이름
            limit (int): 가져올 추천 개수
            
        Returns:
            List[Tuple[int, float]]: (상품 ID, 점수) 형태의 추천 리스트
        """
        try:
            query = """
                SELECT product_id, score
                FROM model_recommendations
                WHERE member_id = :user_id 
                AND model_name = :model_name
                ORDER BY score DESC
                LIMIT :limit
            """
            
            with self.db.get_connection() as conn:
                result = conn.execute(
                    text(query),
                    {
                        'user_id': user_id,
                        'model_name': model_name,
                        'limit': limit
                    }
                )
                
                recommendations = [(row.product_id, float(row.score)) for row in result]
                return recommendations
                
        except Exception as e:
            self.logger.error(f"추천 결과 가져오기 실패: {str(e)}")
            return []
    
    def get_similar_items(self, item_id: int, limit: int = 10) -> List[Tuple[int, float]]:
        """
        특정 아이템과 유사한 아이템을 벡터 데이터베이스에서 가져옵니다.
        
        Args:
            item_id (int): 아이템 ID
            limit (int): 가져올 유사 아이템 개수
            
        Returns:
            List[Tuple[int, float]]: [(아이템 ID, 유사도 점수), ...] 형태의 유사 아이템 리스트
        """
        self.logger.info(f"아이템 {item_id}와 유사한 아이템 가져오기")
        
        try:
            similar_items = self.vector_db.get_similar_products_by_id(
                str(item_id), 
                top_k=limit
            )
            return [(int(id_), float(score)) for id_, score in similar_items]

        except Exception as e:
            self.logger.error(f"유사 아이템 가져오기 실패: {str(e)}")
            return [] 