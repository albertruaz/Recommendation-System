"""
MySQL 데이터베이스 커넥터

MySQL/MariaDB 데이터베이스 연결 및 쿼리 실행을 담당합니다.
"""

import os
from typing import Optional, List, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker

from .base_connector import BaseConnector

class MySQLConnector(BaseConnector):
    """MySQL/MariaDB 데이터베이스 연결을 관리하는 클래스"""
    
    def __init__(self):
        """MySQL 커넥터 초기화"""
        super().__init__()
        
        # 데이터베이스 연결 정보
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
        }
    
    def _setup_engine(self) -> None:
        """SQLAlchemy 엔진을 설정합니다."""
        try:
            use_ssh = self.ssh_config.get('ssh_host') and self.ssh_config.get('ssh_username')
            
            # SSH 터널 설정 (필요한 경우)
            if use_ssh:
                self._setup_ssh_tunnel(self.db_config['host'], self.db_config['port'])
                host = '127.0.0.1'
                port = self._tunnel.local_bind_port
            else:
                host = self.db_config['host']
                port = self.db_config['port']
            
            # MySQL 연결 URL 생성
            db_url = (f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}"
                     f"@{host}:{port}/{self.db_config['database']}")
            
            # 엔진 생성
            self._engine = create_engine(
                db_url,
                pool_size=self.pool_config['pool_size'],
                max_overflow=self.pool_config['max_overflow'],
                pool_timeout=self.pool_config['pool_timeout'],
                pool_recycle=self.pool_config['pool_recycle'],
                pool_pre_ping=True,
                poolclass=QueuePool
            )
            
            # 세션 팩토리 생성
            self._Session = sessionmaker(bind=self._engine)
            
        except Exception as e:
            if self._tunnel and self._tunnel.is_active:
                self._tunnel.close()
            raise Exception(f"MySQL 데이터베이스 엔진 설정 실패: {str(e)}")

    def get_s3_url(self, file_name: str) -> str:
        """S3(또는 CloudFront) 경로 생성"""
        cloudfront_domain = os.getenv('S3_CLOUDFRONT_DOMAIN')
        protocol = "https"
        if not file_name or not cloudfront_domain:
            return None
        return f"{protocol}://{cloudfront_domain}/{file_name}"

    def get_product_data(self, where_condition: str = "1!=1", limit: int = 500, batch_no: int = 0) -> List[Tuple]:
        """상품 데이터를 조회합니다."""
        offset = batch_no * limit

        with self.get_session() as session:
            sql = text(f"""
                SELECT 
                    id,
                    main_image,
                    status,
                    primary_category_id,
                    secondary_category_id
                FROM product
                WHERE 
                    {where_condition}
                LIMIT {limit} OFFSET {offset}
            """)
            result = session.execute(sql)

            products = []
            for row in result.fetchall():
                products.append((
                    row[0],  # id
                    self.get_s3_url(row[1]) if row[1] else None,  # main_image -> S3 URL
                    row[2],  # status
                    row[3],  # primary_category_id
                    row[4],  # secondary_category_id
                ))
            return products

    def get_product_url(self, where_condition: str = "1!=1", limit: int = 5000, batch_no: int = 0) -> List[Tuple]:
        """상품 URL을 조회합니다."""
        offset = batch_no * limit

        with self.get_session() as session:
            sql = text(f"""
                SELECT 
                    id,
                    main_image
                FROM product
                WHERE
                {where_condition}
                LIMIT {limit} OFFSET {offset}
            """)
            result = session.execute(sql)

            products = []
            for row in result.fetchall():
                products.append((
                    row[0],  # id
                    self.get_s3_url(row[1]) if row[1] else None,
                ))
            return products
    
    def get_product_ids_by_condition(self, where_condition: str = "1!=1") -> List[int]:
        """조건에 맞는 상품 ID 목록을 조회합니다."""
        with self.get_session() as session:
            sql = text(f"""
                SELECT 
                    id
                FROM product
                WHERE
                {where_condition}
            """)
            result = session.execute(sql).fetchall()
            return [row[0] for row in result]

    def find_links_by_id(self, product_id: str) -> List[Tuple]:
        """특정 상품 ID의 링크 정보를 조회합니다."""
        with self.get_session() as session:
            sql = text("""
                SELECT 
                    id,
                    main_image
                FROM product
                WHERE id = :product_id
            """)
            result = session.execute(sql, {'product_id': product_id})
            
            products = []
            for row in result.fetchall():
                products.append((
                    row[0],  # id
                    self.get_s3_url(row[1]) if row[1] else None,  # main_image -> S3 URL
                ))
            return products 