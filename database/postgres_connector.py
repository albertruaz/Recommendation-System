"""
PostgreSQL 데이터베이스 커넥터

PostgreSQL 데이터베이스 연결 및 벡터 검색 기능을 제공합니다.
PGVector 확장을 통한 벡터 검색 기능을 지원합니다.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker

from .base_connector import BaseConnector

class PostgresConnector(BaseConnector):
    """PostgreSQL 데이터베이스 연결 및 벡터 검색 기능을 제공하는 클래스"""
    
    def __init__(self):
        """PostgreSQL 커넥터 초기화"""
        super().__init__()
        
        # 데이터베이스 연결 정보
        self.db_config = {
            'host': os.getenv('PG_HOST'),
            'port': int(os.getenv('PG_PORT', 5432)),
            'user': os.getenv('PG_USER'),
            'password': os.getenv('PG_PASSWORD'),
            'database': os.getenv('PG_DB_NAME'),
        }
        
        # 연결 풀 설정
        self.pool_config.update({
            'pool_size': int(os.getenv('PG_POOL_SIZE', 5)),
            'max_overflow': int(os.getenv('PG_MAX_OVERFLOW', 10)),
            'pool_timeout': int(os.getenv('PG_POOL_TIMEOUT', 30)),
            'pool_recycle': int(os.getenv('PG_POOL_RECYCLE', 3600)),
        })
    
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
            
            # PostgreSQL 연결 URL 생성
            db_url = (f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}"
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
            raise Exception(f"PostgreSQL 데이터베이스 엔진 설정 실패: {str(e)}")
    
    def fetch_product_ids(self) -> List[int]:
        """
        product 테이블에서 모든 product_id를 가져오는 함수
        
        Returns:
            List[int]: 상품 ID 목록
        """
        with self.get_session() as session:
            query = text("SELECT id FROM product")
            product_ids = session.execute(query).fetchall()
            return [row[0] for row in product_ids]
            
    def upsert_embeddings(self, embeddings: List[Dict]) -> None:
        """
        상품 임베딩 데이터를 삽입하거나 업데이트합니다.
        
        Args:
            embeddings: 임베딩 데이터 목록 (각 항목은 product_id, image_vector, status 등 포함)
        """
        with self.get_session() as session:
            try:
                for item in embeddings:
                    product_id = item["product_id"]
                    image_vector = item["image_vector"]
                    status = item["status"]
                    primary_category_id = item.get("primary_category_id")
                    secondary_category_id = item.get("secondary_category_id")

                    # 벡터 데이터를 PostgreSQL VECTOR 형식으로 변환
                    if not all(isinstance(val, (int, float)) for val in image_vector):
                        raise ValueError(f"Vector contains non-numeric values: {image_vector}")
                    vector_str = "[" + ",".join(map(str, image_vector)) + "]"

                    # SQL 쿼리 실행 (UPSERT)
                    sql = text("""
                        INSERT INTO product (id, status, primary_category_id, secondary_category_id, image_vector)
                        VALUES (:pid, :status, :primary_cat, :secondary_cat, :vec)
                        ON CONFLICT (id)
                        DO UPDATE SET 
                            status = EXCLUDED.status,
                            primary_category_id = EXCLUDED.primary_category_id,
                            secondary_category_id = EXCLUDED.secondary_category_id,
                            image_vector = EXCLUDED.image_vector;
                    """)

                    session.execute(sql, {
                        "pid": product_id,
                        "status": status,
                        "primary_cat": primary_category_id,
                        "secondary_cat": secondary_category_id,
                        "vec": vector_str
                    })
                
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    def get_similar_products(self, product_ids: List[int], top_k: int = 100) -> Dict[int, List[int]]:
        """
        여러 상품에 대해 유사한 상품 목록을 반환합니다.
        
        Args:
            product_ids: 상품 ID 목록
            top_k: 각 상품마다 반환할 유사 상품 수
            
        Returns:
            Dict[int, List[int]]: 상품 ID를 키로, 유사 상품 ID 목록을 값으로 하는 딕셔너리
        """
        if not product_ids:
            return {}
            
        with self.get_session() as session:
            sim_sql = text("""
                WITH ranked AS (
                    SELECT 
                        p1.id AS product_id,
                        p2.id AS similar_id,
                        (p1.image_vector <#> p2.image_vector) AS distance,
                        ROW_NUMBER() OVER (
                            PARTITION BY p1.id 
                            ORDER BY (p1.image_vector <#> p2.image_vector)
                        ) AS rn
                    FROM product p1
                    JOIN product p2 ON
                        p1.id != p2.id
                        AND p1.primary_category_id = p2.primary_category_id
                        AND p1.secondary_category_id = p2.secondary_category_id
                    WHERE
                        p1.id IN :pids
                        AND p2.status = 'SALE'
                )
                SELECT product_id, similar_id, distance
                FROM ranked
                WHERE rn <= :top_k
                ORDER BY product_id, distance
            """)
            rows = session.execute(sim_sql, {"pids": tuple(product_ids), "top_k": top_k}).fetchall()
            
            # 결과를 Dict 형태로 변환
            product_similars = {}
            for product_id, similar_id, distance in rows:
                if product_id not in product_similars:
                    product_similars[product_id] = []
                product_similars[product_id].append(similar_id)

            return product_similars

    def get_similar_products_by_id(self, product_id: int, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        특정 상품과 유사한 상품 목록을 반환합니다.
        
        Args:
            product_id: 상품 ID
            top_k: 반환할 유사 상품 수
            
        Returns:
            List[Tuple[int, float]]: [(상품ID, 거리), ...] 형태의 유사 상품 목록
        """
        with self.get_session() as session:
            # 1) 대상 product_id의 벡터 가져오기
            query_vec_sql = text("SELECT image_vector FROM product WHERE id=:pid")
            res = session.execute(query_vec_sql, {"pid": product_id}).fetchone()
            if not res:
                return []

            target_vec = res[0]

            # 2) 유사도 계산 (코사인 거리)
            sim_sql = text("""
                SELECT id, (image_vector <#> :tvec) AS distance
                FROM product
                WHERE id != :pid
                ORDER BY image_vector <#> :tvec
                LIMIT :top_k
            """)
            rows = session.execute(sim_sql, {"tvec": target_vec, "pid": product_id, "top_k": top_k}).fetchall()

            # [(product_id, distance), ...] 형태로 반환
            return [(row[0], row[1]) for row in rows]
            
    def get_product_vector(self, product_id: int) -> Optional[np.ndarray]:
        """
        지정된 상품 ID의 벡터를 가져옵니다.
        
        Args:
            product_id: 상품 ID
            
        Returns:
            Optional[np.ndarray]: 상품 벡터 (없으면 None)
        """
        with self.get_session() as session:
            query = text("SELECT image_vector FROM product WHERE id = :pid")
            result = session.execute(query, {"pid": product_id}).fetchone()
            if result:
                return self._convert_to_numpy(result[0])
            return None
            
    def get_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        여러 상품 ID의 벡터를 가져옵니다.
        
        Args:
            product_ids: 상품 ID 리스트
            
        Returns:
            Dict[int, np.ndarray]: {상품ID: 벡터} 형태의 딕셔너리
        """
        if not product_ids:
            return {}
            
        with self.get_session() as session:
            query = text("SELECT id, image_vector FROM product WHERE id IN :pids")
            rows = session.execute(query, {"pids": tuple(product_ids)}).fetchall()
            
            result = {}
            for product_id, vector in rows:
                try:
                    result[product_id] = self._convert_to_numpy(vector)
                except Exception as e:
                    print(f"경고: 상품 {product_id}의 벡터 변환 중 오류 발생: {str(e)}")
                
            return result
            
    def search_by_vector(self, query_vector: Union[np.ndarray, List[float]], top_k: int = 100, 
                        exclude_ids: List[int] = None) -> List[Tuple[int, float]]:
        """
        벡터로 유사한 상품을 검색합니다.
        
        Args:
            query_vector: 검색 벡터 (numpy 배열 또는 리스트)
            top_k: 반환할 최대 상품 수
            exclude_ids: 제외할 상품 ID 리스트
            
        Returns:
            List[Tuple[int, float]]: [(상품ID, 거리), ...] 형태의 리스트
        """
        with self.get_session() as session:
            try:
                # 벡터를 PostgreSQL 형식으로 변환
                vector_str = self._convert_to_pg_vector(query_vector)
                
                # 쿼리 구성
                query = """
                    SELECT id, (image_vector <#> :query_vec) AS distance
                    FROM product
                    WHERE status = 'SALE'
                """
                
                # 제외할 ID가 있으면 조건 추가
                params = {"query_vec": vector_str, "top_k": top_k}
                if exclude_ids and len(exclude_ids) > 0:
                    query += " AND id NOT IN :exclude_ids"
                    params["exclude_ids"] = tuple(exclude_ids)
                    
                query += " ORDER BY distance LIMIT :top_k"
                
                # 쿼리 실행
                rows = session.execute(text(query), params).fetchall()
                
                # 결과 반환
                return [(row[0], row[1]) for row in rows]
            except Exception as e:
                print(f"벡터 검색 중 오류 발생: {str(e)}")
                return []
    
    def _convert_to_numpy(self, vector) -> np.ndarray:
        """
        PostgreSQL 벡터를 numpy 배열로 변환합니다.
        
        Args:
            vector: PostgreSQL 벡터 (문자열 또는 리스트 형태)
            
        Returns:
            np.ndarray: 변환된 numpy 배열
        """
        # 문자열 형태의 벡터인 경우 처리 (예: '[1,2,3]')
        if isinstance(vector, str):
            if vector.startswith('[') and vector.endswith(']'):
                vector = vector[1:-1]
            vector_values = [float(x.strip()) for x in vector.split(',')]
            return np.array(vector_values, dtype=np.float32)
        # 이미 배열 형태인 경우
        elif hasattr(vector, '__iter__'):
            return np.array(vector, dtype=np.float32)
        else:
            raise TypeError(f"지원하지 않는 벡터 타입: {type(vector)}")
    
    def _convert_to_pg_vector(self, vector) -> str:
        """
        numpy 배열이나 리스트를 PostgreSQL 벡터 형식으로 변환합니다.
        
        Args:
            vector: numpy 배열 또는 리스트
            
        Returns:
            str: PostgreSQL 벡터 문자열 (예: "[1,2,3]")
        """
        # 입력 벡터 타입 확인 및 변환
        if isinstance(vector, str):
            # 문자열 형태인 경우 파싱
            if vector.startswith('[') and vector.endswith(']'):
                return vector  # 이미 올바른 형식
            vector_list = [float(x.strip()) for x in vector.split(',')]
        elif isinstance(vector, np.ndarray):
            # NumPy 배열은 리스트로 변환
            vector_list = vector.tolist()
        elif isinstance(vector, list):
            # 이미 리스트인 경우
            vector_list = vector
        else:
            # 다른 타입인 경우 변환 시도
            vector_list = list(vector)
        
        # 벡터를 PostgreSQL 형식으로 변환
        return "[" + ",".join(map(str, vector_list)) + "]" 