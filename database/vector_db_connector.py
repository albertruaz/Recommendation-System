import os
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
load_dotenv()

class VectorDBConnector:
    """
    PostgreSQL + PGVector 전용 DBConnector
    """
    def __init__(self):
        self.ssh_host = os.getenv('PG_SSH_HOST')           # SSH가 필요하다면
        self.ssh_username = os.getenv('PG_SSH_USERNAME')
        self.ssh_pkey_path = os.getenv('PG_SSH_PKEY_PATH')
        self.pg_host = os.getenv('PG_HOST')               # PostgreSQL 호스트
        self.pg_port = int(os.getenv('PG_PORT', 5432))    # PostgreSQL 포트
        self.pg_user = os.getenv('PG_USER')
        self.pg_password = os.getenv('PG_PASSWORD')
        self.pg_dbname = os.getenv('PG_DB_NAME')

        # 커넥션 풀 설정값
        self.pool_size = int(os.getenv('PG_POOL_SIZE', 5))
        self.max_overflow = int(os.getenv('PG_MAX_OVERFLOW', 10))
        self.pool_timeout = int(os.getenv('PG_POOL_TIMEOUT', 30))
        self.pool_recycle = int(os.getenv('PG_POOL_RECYCLE', 3600))

        self.tunnel = None
        self.engine = None
        self.Session = None

        # 커넥션 초기화
        self.connect()

    def connect(self):
        # SSH 터널이 필요한 경우
        # (SSH 터널이 없다면 바로 포트/호스트로 연결)
        if self.ssh_host and self.ssh_username and self.ssh_pkey_path:
            self.tunnel = SSHTunnelForwarder(
                (self.ssh_host, 22),
                ssh_username=self.ssh_username,
                ssh_pkey=self.ssh_pkey_path,
                remote_bind_address=(self.pg_host, self.pg_port)
            )
            self.tunnel.start()
            local_port = self.tunnel.local_bind_port
            db_host = '127.0.0.1'
            db_port = local_port
        else:
            # SSH 터널 없이 직접 연결
            db_host = self.pg_host
            db_port = self.pg_port

        db_url = f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}@{db_host}:{db_port}/{self.pg_dbname}"

        self.engine = create_engine(
            db_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle
        )
        self.Session = sessionmaker(bind=self.engine)

    def close(self):
        if self.Session:
            self.Session.close_all()
            self.Session = None
        if self.tunnel and self.tunnel.is_active:
            self.tunnel.close()
        self.tunnel = None
        self.engine = None
        VectorDBConnector._instance = None

    def create_vector_table(self, dimension: int = 1024):
        """
        PGVector 확장 활성화 및 product_embedding 테이블 생성
        """
        session = self.Session()
        try:
            # 1) PGVector 확장 설치
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()

            # 2) 테이블 생성
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS product (
                id                  BIGINT         PRIMARY KEY,
                status              VARCHAR(255),
                primary_category_id BIGINT,
                secondary_category_id BIGINT,
                image_vector        VECTOR(1024)
            );
            """
            session.execute(text(create_table_sql))
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def fetch_product_ids(self):
        """
        product_embedding 테이블에서 모든 product_id를 가져오는 함수
        """
        session = self.Session()
        try:
            query = text("SELECT id FROM product")
            product_ids = session.execute(query).fetchall()
            return [row[0] for row in product_ids]  # 리스트 형태로 반환
        finally:
            session.close()
            
    def upsert_embeddings(self, embeddings: list):
        session = self.Session()
        try:
            for item in embeddings:
                product_id = item["product_id"]
                image_vector = item["image_vector"]
                status = item["status"]
                primary_category_id = item["primary_category_id"]
                secondary_category_id = item["secondary_category_id"]

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
        finally:
            session.close()

    def get_similar_products(self, product_ids: List[str], top_k: int = 100) -> Dict[str, List[str]]:
        if not product_ids:
            return {}
        session = self.Session()
        try:
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

        finally:
            session.close()

    def get_similar_products_by_id(self, product_id: str, top_k: int = 100) -> list:
        """
        예시로 Euclidean distance 사용 (<->)
        Cosine distance를 사용하려면 (<#>) 또는 다른 문법 사용
        """
        session = self.Session()
        try:
            # 1) 대상 product_id의 벡터 가져오기
            query_vec_sql = text("SELECT image_vector FROM product WHERE id=:pid")
            res = session.execute(query_vec_sql, {"pid": product_id}).fetchone()
            if not res:
                return []

            target_vec = res[0]

            # 2) 유사도 계산 (Euclidean distance)
            #    자신 제외, distance ASC로 정렬
            sim_sql = text("""
                SELECT id, (image_vector <#> :tvec) AS distance
                FROM product
                WHERE id != :pid
                ORDER BY image_vector <#> :tvec
                LIMIT :top_k
            """)
            rows = session.execute(sim_sql, {"tvec": target_vec, "pid": product_id, "top_k": top_k}).fetchall()

            # [(product_id, distance), ...] 형태로 반환
            similar_list = [(r[0], r[1]) for r in rows]
            return similar_list
        finally:
            session.close()
            
    def get_product_vector(self, product_id: int):
        """
        지정된 상품 ID의 벡터를 가져옵니다.
        
        Args:
            product_id: 상품 ID
            
        Returns:
            상품 벡터 (없으면 None)
        """
        session = self.Session()
        try:
            query = text("SELECT image_vector FROM product WHERE id = :pid")
            result = session.execute(query, {"pid": product_id}).fetchone()
            if result:
                return result[0]
            return None
        finally:
            session.close()
            
    def get_product_vectors(self, product_ids: List[int]):
        """
        여러 상품 ID의 벡터를 가져옵니다.
        
        Args:
            product_ids: 상품 ID 리스트
            
        Returns:
            {상품ID: 벡터} 형태의 딕셔너리 (벡터는 numpy 배열로 변환됨)
        """
        if not product_ids:
            return {}
            
        session = self.Session()
        try:
            query = text("SELECT id, image_vector FROM product WHERE id IN :pids")
            rows = session.execute(query, {"pids": tuple(product_ids)}).fetchall()
            
            result = {}
            for product_id, vector in rows:
                # PostgreSQL vector 데이터를 NumPy 배열로 변환
                try:
                    # 문자열 형태의 벡터인 경우 처리 (예: '[1,2,3]')
                    if isinstance(vector, str):
                        if vector.startswith('[') and vector.endswith(']'):
                            vector = vector[1:-1]
                        vector_values = [float(x.strip()) for x in vector.split(',')]
                        result[product_id] = np.array(vector_values, dtype=np.float32)
                    # 이미 배열 형태인 경우
                    elif hasattr(vector, '__iter__'):
                        result[product_id] = np.array(vector, dtype=np.float32)
                    else:
                        print(f"경고: 상품 {product_id}의 벡터 형식을، 처리할 수 없습니다. 타입: {type(vector)}")
                        continue
                except Exception as e:
                    print(f"경고: 상품 {product_id}의 벡터 변환 중 오류 발생: {str(e)}")
                    continue
                
            return result
        finally:
            session.close()
            
    def search_by_vector(self, query_vector, top_k: int = 100, exclude_ids: List[int] = None):
        """
        벡터로 유사한 상품을 검색합니다.
        
        Args:
            query_vector: 검색 벡터 (numpy 배열 또는 리스트)
            top_k: 반환할 최대 상품 수
            exclude_ids: 제외할 상품 ID 리스트
            
        Returns:
            [(product_id, distance), ...] 형태의 리스트
        """
        session = self.Session()
        try:
            # 입력 벡터 타입 확인 및 변환
            if isinstance(query_vector, str):
                # 문자열 형태인 경우 파싱
                if query_vector.startswith('[') and query_vector.endswith(']'):
                    query_vector = query_vector[1:-1]
                query_vector = [float(x.strip()) for x in query_vector.split(',')]
            elif isinstance(query_vector, np.ndarray):
                # NumPy 배열은 리스트로 변환
                query_vector = query_vector.tolist()
            elif not isinstance(query_vector, list):
                # 리스트가 아닌 경우 변환 시도
                query_vector = list(query_vector)
            
            # 벡터를 PostgreSQL 형식으로 변환
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
            
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
        finally:
            session.close() 