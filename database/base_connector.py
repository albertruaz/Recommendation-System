"""
기본 데이터베이스 커넥터 추상 클래스

모든 데이터베이스 커넥터의 공통 인터페이스와 기능을 정의합니다.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

class BaseConnector(ABC):
    """데이터베이스 연결을 관리하는 추상 클래스"""
    
    def __init__(self):
        """커넥터 초기화"""
        load_dotenv()
        self._engine: Optional[Engine] = None
        self._Session = None
        self._tunnel: Optional[SSHTunnelForwarder] = None
        
        # SSH 터널링 정보
        self.ssh_config = {
            'ssh_host': os.getenv('SSH_HOST'),
            'ssh_username': os.getenv('SSH_USERNAME'),
            'ssh_pkey_path': os.getenv('SSH_PKEY_PATH'),
        }
        
        # 연결 풀 설정
        self.pool_config = {
            'pool_size': int(os.getenv('POOL_SIZE', 5)),
            'max_overflow': int(os.getenv('MAX_OVERFLOW', 10)),
            'pool_timeout': int(os.getenv('POOL_TIMEOUT', 30)),
            'pool_recycle': int(os.getenv('POOL_RECYCLE', 3600)),
        }
    
    def _setup_ssh_tunnel(self, remote_host, remote_port) -> None:
        """SSH 터널을 설정합니다."""
        try:
            if self._tunnel is None or not self._tunnel.is_active:
                self._tunnel = SSHTunnelForwarder(
                    (self.ssh_config['ssh_host'], 22),
                    ssh_username=self.ssh_config['ssh_username'],
                    ssh_pkey=self.ssh_config['ssh_pkey_path'],
                    remote_bind_address=(remote_host, remote_port),
                    local_bind_address=('127.0.0.1', 0)  # 동적 로컬 포트 할당
                )
                self._tunnel.start()
        except Exception as e:
            raise Exception(f"SSH 터널 설정 실패: {str(e)}")
    
    @abstractmethod
    def _setup_engine(self) -> None:
        """SQLAlchemy 엔진을 설정합니다. 구체적인 구현은 서브클래스에서 정의해야 합니다."""
        pass
    
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