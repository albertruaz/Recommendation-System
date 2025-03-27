"""
추천 시스템과 데이터베이스 연결을 위한 유틸리티 모듈

이 모듈은 데이터베이스에서 사용자-아이템 상호작용 데이터를 가져오고,
추천 모델에 필요한 형식으로 변환하는 기능을 제공합니다.
"""

import os
import logging
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
from .db_connector import DBConnector
from sqlalchemy import text
from utils.config import load_config

class RecommendationDB:
    """추천 시스템을 위한 데이터베이스 유틸리티 클래스"""
    
    def __init__(self):
        """데이터베이스 연결 초기화"""
        self.db = DBConnector()
        self.logger = logging.getLogger(__name__)
        self.config = load_config()
    
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