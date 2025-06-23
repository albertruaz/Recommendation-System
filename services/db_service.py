"""
데이터베이스 관련 통합 서비스 (로딩 + 저장)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from database.db_manager import DatabaseManager
from database.db import db
from utils.logger import setup_logger
from sqlalchemy import text


class DatabaseService:
    """데이터베이스 로딩과 저장을 통합 처리하는 서비스"""
    
    def __init__(self, db_type: str = "postgres", interaction_weights: Dict[str, float] = None):
        self.db_type = db_type
        self.db_manager = DatabaseManager()
        self.interaction_weights = interaction_weights or {}
        self.logger = setup_logger('db_service')
        
        # 인덱스 매핑 (DataLoader에서 이동)
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
    
    # DataLoader 기능들을 통합
    def load_interactions(self, days: int = 30) -> pd.DataFrame:
        db_instance = db()
        interactions = db_instance.get_user_item_interactions(days=days, use_cache=True)
        return interactions
    
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

        df = interactions_df.copy()

        df['rating'] = df['interaction_type'].map(self.interaction_weights)
        df['user_idx'] = df['member_id'].map(self.user2idx)
        df['item_idx'] = df['product_id'].map(self.item2idx)
        
        return df[['user_idx', 'item_idx', 'rating']]
    
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
            with connector.get_session() as session:
                # 테이블 생성 (존재하지 않는 경우)
                create_table_sql = self._get_create_recommendations_table_sql()
                session.execute(text(create_table_sql))
                
                # 데이터 삽입
                insert_sql = self._get_insert_recommendations_sql()
                for _, row in grouped_recommendations.iterrows():
                    session.execute(text(insert_sql), {
                        'member_id': int(row['member_id']),
                        'product_ids': row['product_ids_json'],
                        'created_at': row['created_at']
                    })
            
            self.logger.info(f"추천 결과 저장 완료: {len(grouped_recommendations)}명 사용자")
            
            return {
                'table_name': 'recommendations',
                'record_count': len(grouped_recommendations),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"추천 결과 저장 중 오류: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def save_recommendations_to_db(self, recommendations_df: pd.DataFrame, 
                                  run_id: str, table_name: str = "recommendations") -> Dict:
        """추천 결과를 데이터베이스에 저장"""
        try:
            # run_id와 타임스탬프 추가
            data_to_save = recommendations_df.copy()
            data_to_save['run_id'] = run_id
            data_to_save['created_at'] = pd.Timestamp.now()
            
            # 컬럼 순서 정리
            column_order = ['run_id', 'member_id', 'product_id', 'predicted_rating', 'created_at']
            data_to_save = data_to_save[column_order]
            
            # 데이터베이스 커넥터 선택
            if self.db_type == "postgres":
                connector = self.db_manager.postgres
            elif self.db_type == "mysql":
                connector = self.db_manager.mysql
            else:
                raise ValueError(f"지원하지 않는 데이터베이스 타입: {self.db_type}")
            
            # 실제 저장 구현
            with connector.get_session() as session:
                # 테이블 생성 (존재하지 않는 경우)
                create_table_sql = self._get_create_table_sql(table_name)
                session.execute(text(create_table_sql))
                
                # 데이터 삽입
                insert_sql = self._get_insert_sql(table_name)
                for _, row in data_to_save.iterrows():
                    session.execute(text(insert_sql), {
                        'run_id': row['run_id'],
                        'member_id': int(row['member_id']),
                        'product_id': int(row['product_id']),
                        'predicted_rating': float(row['predicted_rating']),
                        'created_at': row['created_at']
                    })
            
            self.logger.info(f"추천 결과 DB 저장 완료: {table_name} ({len(data_to_save)}개 레코드)")
            
            return {
                'table_name': table_name,
                'record_count': len(data_to_save),
                'run_id': run_id,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"DB 저장 중 오류: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def save_user_factors_to_db(self, user_factors_df: pd.DataFrame, 
                               run_id: str, table_name: str = "user_factors") -> Dict:
        """사용자 잠재 요인을 데이터베이스에 저장"""
        try:
            data_to_save = user_factors_df.copy()
            data_to_save['run_id'] = run_id
            data_to_save['created_at'] = pd.Timestamp.now()
            
            connector = self.db_manager.postgres if self.db_type == "postgres" else self.db_manager.mysql
            
            with connector.get_session() as session:
                # 테이블 생성 (존재하지 않는 경우)
                create_table_sql = self._get_create_user_factors_table_sql(table_name)
                session.execute(text(create_table_sql))
                
                # 데이터 삽입
                insert_sql = self._get_insert_user_factors_sql(table_name)
                for _, row in data_to_save.iterrows():
                    features_str = ','.join(map(str, row['features']))
                    session.execute(text(insert_sql), {
                        'run_id': row['run_id'],
                        'user_id': int(row['id']),
                        'features': features_str,
                        'created_at': row['created_at']
                    })
            
            self.logger.info(f"사용자 요인 DB 저장 완료: {table_name} ({len(data_to_save)}개 레코드)")
            
            return {
                'table_name': table_name,
                'record_count': len(data_to_save),
                'run_id': run_id,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"사용자 요인 DB 저장 중 오류: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def save_item_factors_to_db(self, item_factors_df: pd.DataFrame, 
                               run_id: str, table_name: str = "item_factors") -> Dict:
        """아이템 잠재 요인을 데이터베이스에 저장"""
        try:
            data_to_save = item_factors_df.copy()
            data_to_save['run_id'] = run_id
            data_to_save['created_at'] = pd.Timestamp.now()
            
            connector = self.db_manager.postgres if self.db_type == "postgres" else self.db_manager.mysql
            
            with connector.get_session() as session:
                # 테이블 생성 (존재하지 않는 경우)
                create_table_sql = self._get_create_item_factors_table_sql(table_name)
                session.execute(text(create_table_sql))
                
                # 데이터 삽입
                insert_sql = self._get_insert_item_factors_sql(table_name)
                for _, row in data_to_save.iterrows():
                    features_str = ','.join(map(str, row['features']))
                    session.execute(text(insert_sql), {
                        'run_id': row['run_id'],
                        'item_id': int(row['id']),
                        'features': features_str,
                        'created_at': row['created_at']
                    })
            
            self.logger.info(f"아이템 요인 DB 저장 완료: {table_name} ({len(data_to_save)}개 레코드)")
            
            return {
                'table_name': table_name,
                'record_count': len(data_to_save),
                'run_id': run_id,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"아이템 요인 DB 저장 중 오류: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_create_table_sql(self, table_name: str) -> str:
        """추천 결과 테이블 생성 SQL 반환"""
        if self.db_type == "postgres":
            return f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255) NOT NULL,
                    member_id BIGINT NOT NULL,
                    product_id BIGINT NOT NULL,
                    predicted_rating DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    INDEX idx_run_id (run_id),
                    INDEX idx_member_id (member_id),
                    INDEX idx_product_id (product_id)
                )
            """
        else:  # mysql
            return f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    run_id VARCHAR(255) NOT NULL,
                    member_id BIGINT NOT NULL,
                    product_id BIGINT NOT NULL,
                    predicted_rating DOUBLE NOT NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_run_id (run_id),
                    INDEX idx_member_id (member_id),
                    INDEX idx_product_id (product_id)
                )
            """
    
    def _get_insert_sql(self, table_name: str) -> str:
        """추천 결과 삽입 SQL 반환"""
        return f"""
            INSERT INTO {table_name} (run_id, member_id, product_id, predicted_rating, created_at)
            VALUES (:run_id, :member_id, :product_id, :predicted_rating, :created_at)
        """
    
    def _get_create_user_factors_table_sql(self, table_name: str) -> str:
        """사용자 요인 테이블 생성 SQL 반환"""
        if self.db_type == "postgres":
            return f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255) NOT NULL,
                    user_id BIGINT NOT NULL,
                    features TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    INDEX idx_run_id (run_id),
                    INDEX idx_user_id (user_id)
                )
            """
        else:  # mysql
            return f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    run_id VARCHAR(255) NOT NULL,
                    user_id BIGINT NOT NULL,
                    features TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_run_id (run_id),
                    INDEX idx_user_id (user_id)
                )
            """
    
    def _get_insert_user_factors_sql(self, table_name: str) -> str:
        """사용자 요인 삽입 SQL 반환"""
        return f"""
            INSERT INTO {table_name} (run_id, user_id, features, created_at)
            VALUES (:run_id, :user_id, :features, :created_at)
        """
    
    def _get_create_item_factors_table_sql(self, table_name: str) -> str:
        """아이템 요인 테이블 생성 SQL 반환"""
        if self.db_type == "postgres":
            return f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255) NOT NULL,
                    item_id BIGINT NOT NULL,
                    features TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    INDEX idx_run_id (run_id),
                    INDEX idx_item_id (item_id)
                )
            """
        else:  # mysql
            return f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    run_id VARCHAR(255) NOT NULL,
                    item_id BIGINT NOT NULL,
                    features TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_run_id (run_id),
                    INDEX idx_item_id (item_id)
                )
            """
    
    def _get_insert_item_factors_sql(self, table_name: str) -> str:
        """아이템 요인 삽입 SQL 반환"""
        return f"""
            INSERT INTO {table_name} (run_id, item_id, features, created_at)
            VALUES (:run_id, :item_id, :features, :created_at)
        """
    
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
    
    def _get_insert_recommendations_sql(self) -> str:
        """추천 결과 삽입 SQL 반환"""
        return """
            INSERT INTO recommendations (member_id, product_ids, created_at)
            VALUES (:member_id, :product_ids, :created_at)
        """

    def cleanup(self):
        """리소스 정리"""
        if self.db_manager:
            self.db_manager.dispose_all() 