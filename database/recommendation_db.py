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
                WITH user_product_interactions AS (
                    -- 조회 (view) 데이터
                    SELECT 
                        v.member_id,
                        v.product_id,
                        'view' as interaction_type,
                        CASE 
                            WHEN COUNT(DISTINCT v.id) >= 3 THEN 1  -- 3회 이상 조회
                            WHEN COUNT(DISTINCT v.id) >= 1 THEN 2  -- 1회 이상 조회
                            ELSE 3                                 -- 단순 노출
                        END as view_type,
                        v.created_at
                    FROM product_view v
                    WHERE v.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    GROUP BY v.member_id, v.product_id
                    
                    UNION ALL
                    
                    -- 좋아요 (like) 데이터
                    SELECT 
                        l.member_id,
                        l.product_id,
                        'like' as interaction_type,
                        NULL as view_type,
                        l.created_at
                    FROM action_log l
                    WHERE l.event_name = 'LikedEvent'
                    AND l.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    
                    UNION ALL
                    
                    -- 장바구니 (cart) 데이터
                    SELECT 
                        c.member_id,
                        c.product_id,
                        'cart' as interaction_type,
                        NULL as view_type,
                        c.created_at
                    FROM action_log c
                    WHERE c.event_name = 'CartItemPutEvent'
                    AND c.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    
                    UNION ALL
                    
                    -- 구매 (purchase) 데이터
                    SELECT 
                        p.customer_id as member_id,
                        p.archived_id as product_id,
                        'purchase' as interaction_type,
                        NULL as view_type,
                        p.created_at
                    FROM purchased_product p
                    WHERE p.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                )
                SELECT 
                    member_id,
                    product_id,
                    interaction_type,
                    view_type,
                    COUNT(*) as interaction_count
                FROM user_product_interactions
                GROUP BY 
                    member_id, 
                    product_id, 
                    interaction_type,
                    view_type
                HAVING interaction_count > 0
                ORDER BY 
                    member_id,
                    product_id,
                    FIELD(interaction_type, 'purchase', 'cart', 'like', 'view'),
                    view_type
            """
            
            with self.db.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params={'days': days})
                self.logger.info(f"총 {len(df)}개의 상호작용 데이터 로드 완료")
                return df
                
        except Exception as e:
            self.logger.error(f"데이터베이스에서 상호작용 데이터 가져오기 실패: {str(e)}")
            raise 