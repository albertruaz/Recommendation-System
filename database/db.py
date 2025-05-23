"""
추천 시스템과 데이터베이스 연결을 위한 유틸리티 모듈

이 모듈은 데이터베이스에서 사용자-아이템 상호작용 데이터를 가져오고,
추천 모델에 필요한 형식으로 변환하는 기능을 제공합니다.
"""

import pandas as pd
from typing import Optional
from sqlalchemy import text
from utils.logger import setup_logger
import os
from datetime import datetime
from .db_manager import DatabaseManager

class db:
    """추천 시스템을 위한 데이터베이스 유틸리티 클래스"""
    
    def __init__(self):
        """데이터베이스 연결 초기화"""
        self.db_manager = DatabaseManager()
        self.logger = setup_logger('db')
        # 캐시 디렉토리 설정
        self.cache_dir = "cache/interactions"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_user_item_interactions(self, days: int = 1, use_cache: bool = True) -> pd.DataFrame:
        """
        사용자-아이템 상호작용 데이터를 가져옵니다.
        
        Args:
            days (int, optional): 최근 몇 일간의 데이터를 가져올지 지정
            use_cache (bool, optional): 캐시 사용 여부
            
        Returns:
            pd.DataFrame: 상호작용 데이터
        """
        today = datetime.now().strftime("%Y%m%d")
        cache_file = os.path.join(self.cache_dir, f"interactions_{days}days_{today}.csv")
        
        # 캐시 파일이 존재하고 사용 가능한 경우
        if use_cache and os.path.exists(cache_file):
            self.logger.info(f"캐시 파일 {cache_file}에서 데이터를 불러옵니다.")
            try:
                df = pd.read_csv(cache_file)
                df['member_id'] = pd.to_numeric(df['member_id'], errors='coerce').astype('Int64')
                df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce').astype('Int64')
                
                # 데이터 통계 로깅
                self.logger.info(f"고유 사용자 수: {df['member_id'].nunique()}")
                self.logger.info(f"고유 상품 수: {df['product_id'].nunique()}")
                
                # 상호작용 타입별 통계
                interaction_stats = df.groupby('interaction_type').size()
                self.logger.info("\n상호작용 타입별 통계:")
                for interaction_type, count in interaction_stats.items():
                    self.logger.info(f"- {interaction_type}: {count}건")
                
                return df
            except Exception as e:
                self.logger.warning(f"캐시 파일 로드 중 오류 발생: {str(e)}. DB에서 직접 데이터를 가져옵니다.")
                # 캐시 파일 로드 실패 시 DB에서 직접 가져오기로 진행
        
        try:
            self.logger.info(f"DB에서 {days}일간의 상호작용 데이터를 가져옵니다.")
            query = """
                WITH impression AS (
                    SELECT 
                        CAST(member_id AS UNSIGNED) as member_id,
                        CAST(product_id AS UNSIGNED) as product_id,
                        COUNT(*) as impression_count
                    FROM product_impression
                    WHERE member_id IS NOT NULL
                        AND created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    GROUP BY member_id, product_id
                ),
                user_product_interactions AS (
                    -- 조회 (view) 데이터
                    SELECT 
                        CAST(v.member_id AS UNSIGNED) as member_id,
                        CAST(v.product_id AS UNSIGNED) as product_id,
                        CASE 
                            WHEN COUNT(DISTINCT v.id) >= 6 THEN 'view2'  -- 6회 이상 조회
                            WHEN COUNT(DISTINCT v.id) >= 3 THEN 'view1'  -- 3회 이상 조회(6회 미만)
                            ELSE NULL
                        END as interaction_type,
                        v.created_at
                    FROM product_view v
                    WHERE v.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    GROUP BY v.member_id, v.product_id
                    HAVING interaction_type IS NOT NULL
                    
                    UNION ALL
                    
                    -- impression 데이터
                    SELECT 
                        member_id,
                        product_id,
                        CASE 
                            WHEN impression_count >= 6 THEN 'impression2'  -- 6회 이상 노출
                            WHEN impression_count >= 3 THEN 'impression1'  -- 3회 이상 노출(6회 미만)
                            ELSE NULL
                        END as interaction_type,
                        NULL as created_at
                    FROM impression
                    HAVING interaction_type IS NOT NULL
                    
                    UNION ALL
                    
                    -- 좋아요 (like) 데이터
                    SELECT 
                        CAST(l.member_id AS UNSIGNED) as member_id,
                        CAST(l.product_id AS UNSIGNED) as product_id,
                        'like' as interaction_type,
                        l.created_at
                    FROM action_log l
                    WHERE l.event_name = 'LikedEvent'
                    AND l.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    
                    UNION ALL
                    
                    -- 장바구니 (cart) 데이터
                    SELECT 
                        CAST(c.member_id AS UNSIGNED) as member_id,
                        CAST(c.product_id AS UNSIGNED) as product_id,
                        'cart' as interaction_type,
                        c.created_at
                    FROM action_log c
                    WHERE c.event_name = 'CartItemPutEvent'
                    AND c.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    
                    UNION ALL
                    
                    -- 구매 (purchase) 데이터
                    SELECT 
                        CAST(p.customer_id AS UNSIGNED) as member_id,
                        CAST(p.archived_id AS UNSIGNED) as product_id,
                        'purchase' as interaction_type,
                        p.created_at
                    FROM purchased_product p
                    WHERE p.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                )
                SELECT 
                    member_id,
                    product_id,
                    interaction_type
                FROM user_product_interactions
                WHERE member_id IS NOT NULL
                  AND product_id IS NOT NULL
                GROUP BY 
                    member_id, 
                    product_id, 
                    interaction_type
                ORDER BY 
                    member_id,
                    product_id,
                    FIELD(interaction_type, 'purchase', 'cart', 'like', 'view2', 'view1', 'impression2', 'impression1')
            """

            with self.db_manager.mysql.get_connection() as conn:
                start_time = datetime.now()
                df = pd.read_sql(text(query), conn, params={'days': days})
                end_time = datetime.now()
                query_time = (end_time - start_time).total_seconds()
                self.logger.info(f"DB 쿼리 완료. 소요 시간: {query_time:.2f}초")
                
                df['member_id'] = pd.to_numeric(df['member_id'], errors='coerce').astype('Int64')
                df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['member_id', 'product_id'])
                
                # 데이터 통계 로깅
                self.logger.info(f"총 {len(df)}개의 상호작용 데이터 로드 완료")
                self.logger.info(f"고유 사용자 수: {df['member_id'].nunique()}")
                self.logger.info(f"고유 상품 수: {df['product_id'].nunique()}")
                
                # 상호작용 타입별 통계
                interaction_stats = df.groupby('interaction_type').size()
                self.logger.info("\n상호작용 타입별 통계:")
                for interaction_type, count in interaction_stats.items():
                    self.logger.info(f"- {interaction_type}: {count}건")
                
                # 캐시 파일 저장
                if use_cache:
                    try:
                        df.to_csv(cache_file, index=False)
                        self.logger.info(f"상호작용 데이터가 캐시 파일 {cache_file}에 저장되었습니다.")
                    except Exception as e:
                        self.logger.warning(f"캐시 파일 저장 중 오류 발생: {str(e)}")
                
                return df
                
        except Exception as e:
            self.logger.error(f"데이터베이스에서 상호작용 데이터 가져오기 실패: {str(e)}")
            raise 
            
    def get_recent_cart_items(self, days: int = 30, min_interactions: int = 2, max_cart_items: int = 20) -> pd.DataFrame:
        """
        사용자별 최근 구매한 상품 목록과 관련 정보를 가져옵니다.
        
        Args:
            days (int, optional): 최근 몇 일간의 데이터를 가져올지 지정
            min_interactions (int, optional): 사용자별 최소 상품 상호작용 수
            max_cart_items (int, optional): 사용자별 최대 상품 개수 (최신순)
            
        Returns:
            pd.DataFrame: 구매 상품 데이터 (member_id, product_id, primary_category_id, secondary_category_id, styles_id 컬럼 포함)
        """
        cache_file = os.path.join(self.cache_dir, f"purchase_items_{days}days_min{min_interactions}_max{max_cart_items}_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # 캐시 확인
        if os.path.exists(cache_file):
            self.logger.info(f"캐시 파일 {cache_file}에서 구매 데이터를 불러옵니다.")
            try:
                df = pd.read_csv(cache_file)
                
                self.logger.info(f"총 {len(df)}개의 구매 데이터 로드 완료")
                self.logger.info(f"고유 사용자 수: {df['member_id'].nunique()}")
                self.logger.info(f"고유 상품 수: {df['product_id'].nunique()}")
                
                return df
            except Exception as e:
                self.logger.warning(f"캐시 파일 로드 중 오류 발생: {str(e)}. DB에서 직접 데이터를 가져옵니다.")
        
        try:
            self.logger.info(f"DB에서 {days}일간의 구매 데이터를 가져옵니다. (최소 상호작용 수: {min_interactions}, 최대 상품 수: {max_cart_items})")
            query = """
                WITH user_product_counts AS (
                    SELECT 
                        customer_id,
                        COUNT(*) as product_count
                    FROM purchased_product
                    WHERE updated_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    AND customer_id IS NOT NULL
                    GROUP BY customer_id
                    HAVING product_count >= :min_interactions
                ),
                recent_products AS (
                    SELECT 
                        pp.customer_id,
                        pp.archived_id,
                        ROW_NUMBER() OVER (PARTITION BY pp.customer_id ORDER BY pp.updated_at DESC) as row_num
                    FROM purchased_product pp
                    INNER JOIN user_product_counts upc ON pp.customer_id = upc.customer_id
                    WHERE pp.updated_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                    AND pp.customer_id IS NOT NULL
                    AND pp.archived_id IS NOT NULL
                )
                SELECT 
                    CAST(rp.customer_id AS UNSIGNED) as member_id,
                    CAST(rp.archived_id AS UNSIGNED) as product_id,
                    p.primary_category_id,
                    p.secondary_category_id,
                    ps.styles_id
                FROM recent_products rp
                LEFT JOIN product p ON rp.archived_id = p.id
                LEFT JOIN product_styles ps ON p.id = ps.product_id
                WHERE rp.row_num <= :max_cart_items
                ORDER BY rp.customer_id, rp.row_num
            """

            with self.db_manager.mysql.get_connection() as conn:
                start_time = datetime.now()
                df = pd.read_sql(text(query), conn, params={'days': days, 'min_interactions': min_interactions, 'max_cart_items': max_cart_items})
                end_time = datetime.now()
                query_time = (end_time - start_time).total_seconds()
                self.logger.info(f"DB 쿼리 완료. 소요 시간: {query_time:.2f}초")
                
                df['member_id'] = pd.to_numeric(df['member_id'], errors='coerce').astype('Int64')
                df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['member_id', 'product_id'])
                
                # 데이터 통계 로깅
                self.logger.info(f"총 {len(df)}개의 구매 데이터 로드 완료")
                self.logger.info(f"고유 사용자 수: {df['member_id'].nunique()}")
                self.logger.info(f"고유 상품 수: {df['product_id'].nunique()}")
                
                # 캐시 파일 저장
                try:
                    df.to_csv(cache_file, index=False)
                    self.logger.info(f"구매 데이터가 캐시 파일 {cache_file}에 저장되었습니다.")
                except Exception as e:
                    self.logger.warning(f"캐시 파일 저장 중 오류 발생: {str(e)}")
                
                return df
                
        except Exception as e:
            self.logger.error(f"데이터베이스에서 구매 데이터 가져오기 실패: {str(e)}")
            # 오류 발생 시 빈 DataFrame 반환
            return pd.DataFrame(columns=['member_id', 'product_id', 'primary_category_id', 'secondary_category_id', 'styles_id']) 