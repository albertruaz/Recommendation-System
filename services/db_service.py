"""
데이터베이스 관련 통합 서비스 (로딩 + 저장)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from database.db_manager import DatabaseManager
from utils.logger import setup_logger
from sqlalchemy import text


class DatabaseService:
    """데이터베이스 로딩과 저장을 통합 처리하는 서비스"""
    
    def __init__(self, db_type: str = "postgres", interaction_weights: Dict[str, float] = None, interaction_thresholds: Dict[str, int] = None):
        self.db_type = db_type
        self.db_manager = DatabaseManager()
        self.interaction_weights = interaction_weights or {}
        self.interaction_thresholds = interaction_thresholds or {
            'view1': 3, 'view2': 6, 'view3': 10,
            'impression1': 3, 'impression2': 6, 'impression3': 10
        }
        self.logger = setup_logger('db_service')
        
        # 캐시 디렉토리 설정 (db 클래스에서 이동)
        self.cache_dir = "cache/interactions"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 인덱스 매핑 (DataLoader에서 이동)
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
    
    # DataLoader 기능들을 통합 + db 클래스 기능 통합
    def load_interactions(self, days: int = 30) -> pd.DataFrame:
        """db 클래스의 get_user_item_interactions를 직접 구현"""
        return self.get_user_item_interactions(days=days, use_cache=True)
    
    def get_user_item_interactions(self, days: int = 1, use_cache: bool = True) -> pd.DataFrame:
        """
        사용자-아이템 상호작용 데이터를 가져옵니다. (db 클래스에서 통합)
        
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
        
        try:
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
                            WHEN COUNT(DISTINCT v.id) >= :view3_threshold THEN 'view3'
                            WHEN COUNT(DISTINCT v.id) >= :view2_threshold THEN 'view2'
                            WHEN COUNT(DISTINCT v.id) >= :view1_threshold THEN 'view1'
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
                            WHEN impression_count >= :impression3_threshold THEN 'impression3'
                            WHEN impression_count >= :impression2_threshold THEN 'impression2'
                            WHEN impression_count >= :impression1_threshold THEN 'impression1'
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
                    FIELD(interaction_type, 'purchase', 'cart', 'like', 'view3', 'view2', 'view1', 'impression3', 'impression2', 'impression1')
            """

            with self.db_manager.mysql.get_connection() as conn:
                params = {
                    'days': days,
                    'view1_threshold': self.interaction_thresholds.get('view1', 3),
                    'view2_threshold': self.interaction_thresholds.get('view2', 6),
                    'view3_threshold': self.interaction_thresholds.get('view3', 10),
                    'impression1_threshold': self.interaction_thresholds.get('impression1', 3),
                    'impression2_threshold': self.interaction_thresholds.get('impression2', 6),
                    'impression3_threshold': self.interaction_thresholds.get('impression3', 10)
                }
                df = pd.read_sql(text(query), conn, params=params)
                df['member_id'] = pd.to_numeric(df['member_id'], errors='coerce').astype('Int64')
                df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['member_id', 'product_id'])
                
                self.logger.info(f"총 {len(df)}개의 상호작용 데이터 로드 완료")
                self.logger.info(f"고유 사용자 수: {df['member_id'].nunique()}")
                self.logger.info(f"고유 상품 수: {df['product_id'].nunique()}")
                interaction_stats = df.groupby('interaction_type').size()
                self.logger.info("\n상호작용 타입별 통계:")
                for interaction_type, count in interaction_stats.items():
                    self.logger.info(f"- {interaction_type}: {count}건")

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
    
    def prepare_indices(self, interactions_df: pd.DataFrame):
        """사용자 및 상품 ID를 연속적인 인덱스로 매핑"""
        unique_users = interactions_df['member_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        self.user2idx = {user: i for i, user in enumerate(unique_users)}
        self.item2idx = {item: i for i, item in enumerate(unique_items)}
        self.idx2user = {i: user for user, i in self.user2idx.items()}
        self.idx2item = {i: item for item, i in self.item2idx.items()}
        
        self.logger.info(f"사용자: {len(unique_users)}, 상품: {len(unique_items)}")
    
    def transform_to_ratings(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """메모리 효율적인 변환 - 불필요한 복사 방지"""
        # 새 컬럼만 추가하고 필요한 컬럼만 선택
        result = interactions_df.assign(
            rating=interactions_df['interaction_type'].map(self.interaction_weights),
            user_idx=interactions_df['member_id'].map(self.user2idx),
            item_idx=interactions_df['product_id'].map(self.item2idx)
        )[['user_idx', 'item_idx', 'rating']]
        
        return result
    
    def save_recommendations(self, results: Dict) -> Dict:
        """추천 결과를 member_id별로 product_ids 리스트 형태로 저장"""
        try:
            recommendations_df = results['recommendations']
            
            # member_id별로 product_id 리스트 생성
            grouped_recommendations = recommendations_df.groupby('member_id')['product_id'].apply(list).reset_index()
            grouped_recommendations.columns = ['member_id', 'product_ids']
            
            # product_ids를 JSON 문자열로 변환 (DB 저장용)
            grouped_recommendations['product_ids_json'] = grouped_recommendations['product_ids'].apply(
                lambda x: ','.join(map(str, x))
            )
            
            # 메타데이터 추가
            grouped_recommendations['created_at'] = pd.Timestamp.now()
            
            # 데이터베이스 커넥터 선택
            if self.db_type == "postgres":
                connector = self.db_manager.postgres
            elif self.db_type == "mysql":
                connector = self.db_manager.mysql
            else:
                raise ValueError(f"지원하지 않는 데이터베이스 타입: {self.db_type}")
            
            # 데이터베이스에 저장
            engine = connector.engine
            
            # 테이블 생성 (존재하지 않는 경우)
            create_table_sql = self._get_create_recommendations_table_sql()
            with connector.get_session() as session:
                session.execute(text(create_table_sql))
                session.commit()
            
            # pandas to_sql을 사용한 가장 효율적인 저장
            grouped_recommendations[['member_id', 'product_ids_json', 'created_at']].rename(
                columns={'product_ids_json': 'product_ids'}
            ).to_sql(
                name='recommendations',
                con=engine,
                if_exists='append',
                index=False,
                method='multi',  # 배치 처리
                chunksize=1000  # 1000개씩 배치로 처리
            )
            
            self.logger.info(f"추천 결과 저장 완료: {len(grouped_recommendations)}명 사용자")
            
            return {
                'table_name': 'recommendations',
                'record_count': len(grouped_recommendations),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"추천 결과 저장 중 오류: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_create_recommendations_table_sql(self) -> str:
        """추천 결과 테이블 생성 SQL 반환"""
        if self.db_type == "postgres":
            return """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id SERIAL PRIMARY KEY,
                    member_id BIGINT NOT NULL,
                    product_ids TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_recommendations_member_id ON recommendations (member_id);
                CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON recommendations (created_at);
            """
        else:  # mysql
            return """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    member_id BIGINT NOT NULL,
                    product_ids TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_member_id (member_id),
                    INDEX idx_created_at (created_at)
                )
            """
    


    def cleanup(self):
        """리소스 정리"""
        if self.db_manager:
            self.db_manager.dispose_all() 