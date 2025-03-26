"""
데이터베이스 연결을 관리하는 모듈입니다.
"""

import os
from typing import Optional
from contextlib import contextmanager
import logging
import mysql.connector
from mysql.connector import pooling
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

class DBConnector:
    """데이터베이스 연결을 관리하는 클래스"""
    
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        self._engine: Optional[Engine] = None
        self._Session = None
        self._tunnel: Optional[SSHTunnelForwarder] = None
        
        # SSH 터널링 정보
        self.ssh_config = {
            'ssh_host': os.getenv('SSH_HOST'),
            'ssh_username': os.getenv('SSH_USERNAME'),
            'ssh_pkey_path': os.getenv('SSH_PKEY_PATH'),
        }
        
        # 데이터베이스 연결 정보
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
        }
        
        # 연결 풀 설정
        self.pool_config = {
            'pool_size': int(os.getenv('POOL_SIZE', 5)),
            'max_overflow': int(os.getenv('MAX_OVERFLOW', 10)),
            'pool_timeout': int(os.getenv('POOL_TIMEOUT', 30)),
            'pool_recycle': int(os.getenv('POOL_RECYCLE', 3600)),
        }
    
    def _setup_ssh_tunnel(self) -> None:
        """SSH 터널을 설정합니다."""
        try:
            if self._tunnel is None or not self._tunnel.is_active:
                self.logger.info("SSH 터널 설정 시작")
                self._tunnel = SSHTunnelForwarder(
                    (self.ssh_config['ssh_host'], 22),
                    ssh_username=self.ssh_config['ssh_username'],
                    ssh_pkey=self.ssh_config['ssh_pkey_path'],
                    remote_bind_address=(self.db_config['host'], self.db_config['port']),
                    local_bind_address=('127.0.0.1', 0)  # 동적 로컬 포트 할당
                )
                self._tunnel.start()
                self.logger.info(f"SSH 터널 설정 완료 (로컬 포트: {self._tunnel.local_bind_port})")
        except Exception as e:
            self.logger.error(f"SSH 터널 설정 실패: {str(e)}")
            raise
    
    def _setup_engine(self) -> None:
        """SQLAlchemy 엔진을 설정합니다."""
        try:
            # SSH 터널 설정
            self._setup_ssh_tunnel()
            
            # MySQL 연결 URL 생성 (SSH 터널을 통한 연결)
            db_url = (f"mysql+mysqlconnector://{self.db_config['user']}:{self.db_config['password']}"
                     f"@127.0.0.1:{self._tunnel.local_bind_port}/{self.db_config['database']}")
            
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
            
            self.logger.info("데이터베이스 엔진 설정 완료")
            
        except Exception as e:
            self.logger.error(f"데이터베이스 엔진 설정 실패: {str(e)}")
            if self._tunnel and self._tunnel.is_active:
                self._tunnel.close()
            raise
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결을 제공하는 컨텍스트 매니저

        Yields:
            Connection: 데이터베이스 연결 객체
        """
        if not self._engine:
            self._setup_engine()
            
        conn = self._engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def get_session(self):
        """데이터베이스 세션을 제공하는 컨텍스트 매니저

        Yields:
            Session: SQLAlchemy 세션 객체
        """
        if not self._Session:
            self._setup_engine()
            
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def dispose(self):
        """엔진과 연결 풀을 정리합니다."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._Session = None
            
        if self._tunnel and self._tunnel.is_active:
            self._tunnel.close()
            self._tunnel = None
            
        self.logger.info("데이터베이스 연결 및 SSH 터널 정리 완료")

    def get_s3_url(self, file_name: str) -> str:
        """S3(또는 CloudFront) 경로 생성"""
        cloudfront_domain = os.getenv('S3_CLOUDFRONT_DOMAIN')
        protocol = "https"
        if not file_name or not cloudfront_domain:
            return None
        return f"{protocol}://{cloudfront_domain}/{file_name}"

    def get_product_data(self, where_condition: str = "1!=1", limit: int = 500, batch_no: int = 0) -> list:
        offset = batch_no * limit

        session = self._Session()
        try:
            sql = text(f"""
                SELECT 
                    id,
                    main_image,
                    status,
                    primary_category_id,
                    secondary_category_id
                FROM product
                WHERE 
                    -- status LIKE "SALE" AND
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
        finally:
            session.close()

    def get_product_url(self, where_condition: str = "1!=1", limit: int = 5000, batch_no: int = 0) -> list:
        offset = batch_no * limit

        session = self._Session()
        try:
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
        finally:
            session.close()
    
    def get_product_ids_by_condition(self, where_condition: str = "1!=1") -> list:
        session = self._Session()
        try:
            sql = text(f"""
                SELECT 
                    id
                FROM product
                WHERE
                {where_condition}
            """)
            result = session.execute(sql).fetchall()
            return [row[0] for row in result]
        finally:
            session.close()

    def find_links_by_id(self, product_id: str) -> list:
        session = self._Session()
        try:
            sql = text("""
                SELECT link
                FROM product
                WHERE id = :product_id
            """)
            results = session.execute(sql, {"product_id": product_id}).fetchall()
            return [row[0] for row in results if row[0]]
        finally:
            session.close()
