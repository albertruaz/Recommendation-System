"""
데이터베이스 매니저

여러 데이터베이스 커넥터를 중앙에서 관리하고, 필요한 커넥터를 제공합니다.
"""

from typing import Dict, Any, Optional
from .mysql_connector import MySQLConnector
from .postgres_connector import PostgresConnector

class DatabaseManager:
    """여러 데이터베이스 커넥터를 관리하는 매니저 클래스"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """데이터베이스 매니저 초기화"""
        if self._initialized:
            return
            
        self._mysql_connector = None
        self._postgres_connector = None
        self._initialized = True
    
    @property
    def mysql(self) -> MySQLConnector:
        """MySQL 커넥터 인스턴스 반환

        Returns:
            MySQLConnector: MySQL 커넥터 인스턴스
        """
        if self._mysql_connector is None:
            self._mysql_connector = MySQLConnector()
        return self._mysql_connector
    
    @property
    def postgres(self) -> PostgresConnector:
        """PostgreSQL 커넥터 인스턴스 반환

        Returns:
            PostgresConnector: PostgreSQL 커넥터 인스턴스
        """
        if self._postgres_connector is None:
            self._postgres_connector = PostgresConnector()
        return self._postgres_connector
    
    def dispose_all(self):
        """모든 데이터베이스 연결 정리"""
        if self._mysql_connector:
            self._mysql_connector.dispose()
            self._mysql_connector = None
            
        if self._postgres_connector:
            self._postgres_connector.dispose()
            self._postgres_connector = None
    
    def __del__(self):
        """소멸자 - 모든 연결 정리"""
        self.dispose_all() 