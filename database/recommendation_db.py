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
        """
        각 상호작용 타입별 CTE 쿼리를 생성합니다.
        """
        ctes = {}
        rating_rules = self.config['rating_rules']

        # 각 상호작용 타입별 CTE 생성
        for interaction_type, rule in rating_rules.items():
            if interaction_type in ['view', 'impression']:
                # 단순 카운트 테이블
                ctes[interaction_type] = f"""
                    {interaction_type} as (
                        select {rule['user_col']}, {rule['item_col']}, 
                               count(*) as {rule['count_col']}
                        from {rule['table']}
                        where {rule['user_col']} is not null
                            and created_at >= {start_date}
                        group by {rule['user_col']}, {rule['item_col']}
                    )
                """
            elif interaction_type in ['like', 'cart']:
                # 특정 이벤트 필터링이 필요한 테이블
                ctes[interaction_type] = f"""
                    {interaction_type}_action as (
                        select {rule['user_col']}, {rule['item_col']}, 
                               count(*) as {rule['count_col']}
                        from {rule['table']}
                        where {rule['user_col']} is not null
                            and created_at >= {start_date}
                            and event_name = '{rule['event_name']}'
                        group by {rule['user_col']}, {rule['item_col']}
                    )
                """
            elif interaction_type == 'purchase':
                # 구매 테이블
                ctes[interaction_type] = f"""
                    purchase as (
                        select {rule['user_col']}, {rule['item_col']}, 
                               count(*) as {rule['count_col']}
                        from {rule['table']}
                        where {rule['user_col']} is not null
                            and created_at >= {start_date}
                        group by {rule['user_col']}, {rule['item_col']}
                    )
                """

        return ctes

    def _build_case_statement(self) -> str:
        """
        rating 값을 계산하는 CASE 문을 생성합니다.
        """
        rating_rules = self.config['rating_rules']
        case_conditions = []

        # 구매
        if 'purchase' in rating_rules:
            case_conditions.append(
                f"when purchase_count <> 0 then {rating_rules['purchase']['value']}"
            )

        # 장바구니
        if 'cart' in rating_rules:
            case_conditions.append(
                f"when cart_count is not null then {rating_rules['cart']['value']}"
            )

        # 좋아요
        if 'like' in rating_rules:
            case_conditions.append(
                f"when like_count is not null then {rating_rules['like']['value']}"
            )

        # 조회수 기반 점수
        if 'view' in rating_rules:
            for threshold in rating_rules['view']['thresholds']:
                case_conditions.append(
                    f"when view_count >= {threshold['min_count']} then {threshold['value']}"
                )

        # 노출수 기반 점수
        if 'impression' in rating_rules:
            for threshold in rating_rules['impression']['thresholds']:
                case_conditions.append(
                    f"when impression_count >= {threshold['min_count']} then {threshold['value']}"
                )

        case_conditions.append("else 0")
        
        return "case " + " ".join(case_conditions) + " end"

    def _build_where_conditions(self) -> str:
        """
        필터링 조건을 생성합니다.
        """
        filter_conditions = self.config['filter_conditions']
        conditions = []

        # 최소 노출 수 조건
        if 'min_impression_count' in filter_conditions:
            conditions.append(
                f"impression_count >= {filter_conditions['min_impression_count']}"
            )

        # 필수 상호작용 조건
        if 'require_any' in filter_conditions:
            require_conditions = [
                f"{cond} is not null" for cond in filter_conditions['require_any']
            ]
            conditions.append(
                f"({' or '.join(require_conditions)})"
            )

        return " or ".join(conditions)

    def get_user_item_interactions(self, days: Optional[int] = None) -> pd.DataFrame:
        """
        사용자-아이템 상호작용 데이터를 가져옵니다.
        
        Args:
            days (int, optional): 최근 몇 일간의 데이터를 가져올지 지정
            
        Returns:
            pd.DataFrame: 상호작용 데이터
        """
        try:
            if days is None:
                days = self.config['data_preprocessing'].get('default_days', 30)
            
            query = """
                SELECT 
                    i.member_id as member_id,
                    i.product_id as product_id,
                    COUNT(*) as impression_count,
                    COUNT(DISTINCT v.id) as view_count,
                    COUNT(DISTINCT CASE WHEN l.event_name = 'LikedEvent' THEN l.id END) as like_count,
                    COUNT(DISTINCT CASE WHEN c.event_name = 'CartItemPutEvent' THEN c.id END) as cart_count,
                    COUNT(DISTINCT p.id) as purchase_count,
                    CASE
                        WHEN COUNT(DISTINCT p.id) > 0 THEN 5.0
                        WHEN COUNT(DISTINCT CASE WHEN c.event_name = 'CartItemPutEvent' THEN c.id END) > 0 THEN 4.5
                        WHEN COUNT(DISTINCT CASE WHEN l.event_name = 'LikedEvent' THEN l.id END) > 0 THEN 4.0
                        WHEN COUNT(DISTINCT v.id) >= 3 THEN 3.8
                        WHEN COUNT(DISTINCT v.id) >= 1 THEN 3.0
                        WHEN COUNT(*) >= 9 THEN -3.0
                        WHEN COUNT(*) >= 6 THEN -2.0
                        WHEN COUNT(*) >= 3 THEN -1.0
                        ELSE 0
                    END as rating
                FROM product_impression i
                LEFT JOIN product_view v 
                    ON i.member_id = v.member_id
                    AND i.product_id = v.product_id
                    AND v.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                LEFT JOIN action_log l
                    ON i.member_id = l.member_id
                    AND i.product_id = l.product_id
                    AND l.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    AND l.event_name = 'LikedEvent'
                LEFT JOIN action_log c
                    ON i.member_id = c.member_id
                    AND i.product_id = c.product_id
                    AND c.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    AND c.event_name = 'CartItemPutEvent'
                LEFT JOIN purchased_product p
                    ON i.member_id = p.customer_id
                    AND i.product_id = p.archived_id
                    AND p.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                WHERE i.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                GROUP BY i.member_id, i.product_id
                HAVING 
                    impression_count >= 3
                    AND (
                        view_count > 0 OR like_count > 0 
                        OR cart_count > 0 OR purchase_count > 0
                    )
            """
            
            with self.db.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params={'days': days})
                self.logger.info(f"총 {len(df)}개의 상호작용 데이터 로드 완료")
                return df
                
        except Exception as e:
            self.logger.error(f"데이터베이스에서 상호작용 데이터 가져오기 실패: {str(e)}")
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