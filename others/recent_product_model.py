"""
최근 장바구니 상품 기반 추천 모델

사용자가 최근 장바구니에 넣은 상품들의 벡터를 기반으로 유사 상품을 추천합니다.
벡터 DB 연결 및 검색 기능을 직접 구현합니다.

[주의] 이 모델은 레거시 코드로, RunRecentProduct 클래스로 기능이 이전되었습니다.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from database.recommendation_db import db
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv
import warnings

# 환경 변수 로드
load_dotenv()

class RecentProductModel:
    def __init__(self, 
                 top_n: int = 100, 
                 similarity_threshold: float = 0.3,
                 min_interactions: int = 2,
                 max_cart_items: int = 20,
                 use_category_filter: bool = True,
                 include_similar_categories: bool = False):
        """
        최근 장바구니 상품 기반 추천 모델 초기화
        
        [주의] 이 모델은 레거시 코드로, RunRecentProduct 클래스로 기능이 이전되었습니다.
        
        Args:
            top_n: 사용자별 추천할 상품 수
            similarity_threshold: 유사도 임계값 (이 값 이상의 유사도를 가진 상품만 추천)
            min_interactions: 추천을 위한 최소 상호작용 수 
            max_cart_items: 고려할 최대 장바구니 상품 수
            use_category_filter: 카테고리 기반 필터링 사용 여부
            include_similar_categories: 비슷한 카테고리의 상품도 추천에 포함할지 여부
        """
        warnings.warn(
            "RecentProductModel 클래스는 레거시 코드로, RunRecentProduct 클래스로 기능이 이전되었습니다. "
            "대신 run/run_recent_product.py의 RunRecentProduct 클래스를 사용하세요.", 
            DeprecationWarning, 
            stacklevel=2
        )
        
        self.top_n = top_n
        self.similarity_threshold = similarity_threshold
        self.min_interactions = min_interactions
        self.max_cart_items = max_cart_items
        self.use_category_filter = use_category_filter
        self.include_similar_categories = include_similar_categories
        
        self.db = db()
        self.logger = logging.getLogger('recent_product_model')
        
        # 벡터 DB 연결 설정
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

        # 벡터 DB 연결 초기화
        self._connect_vector_db()
        
    def _connect_vector_db(self):
        """벡터 DB(PostgreSQL)에 연결합니다."""
        try:
            # SSH 터널이 필요한 경우
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
            self.logger.info("벡터 DB 연결 성공")
        except Exception as e:
            self.logger.error(f"벡터 DB 연결 실패: {str(e)}")
            raise
            
    def _close_vector_db(self):
        """벡터 DB 연결을 종료합니다."""
        if self.Session:
            self.Session.close_all()
            self.Session = None
        if self.tunnel and self.tunnel.is_active:
            self.tunnel.close()
        self.tunnel = None
        self.engine = None
        
    def get_user_cart_items(self, days: int = 30) -> pd.DataFrame:
        """
        사용자별 최근 장바구니 상품 목록 가져오기
        
        Args:
            days: 최근 몇 일 간의 데이터를 가져올지 설정
            
        Returns:
            사용자별 장바구니 상품 정보가 담긴 DataFrame
        """
        # DB에서 최근 장바구니 상품 정보 가져오기
        cart_items = self.db.get_recent_cart_items(days=days)
        return cart_items
    
    def get_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        상품 ID 목록에 대한 벡터 가져오기
        
        Args:
            product_ids: 상품 ID 목록
            
        Returns:
            상품 ID를 키로, 벡터를 값으로 하는 딕셔너리
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
                        self.logger.warning(f"상품 {product_id}의 벡터 형식을 처리할 수 없습니다. 타입: {type(vector)}")
                        continue
                except Exception as e:
                    self.logger.warning(f"상품 {product_id}의 벡터 변환 중 오류 발생: {str(e)}")
                    continue
                
            return result
        except Exception as e:
            self.logger.error(f"벡터 데이터 조회 중 오류: {str(e)}")
            return {}
        finally:
            session.close()
    
    def compute_average_vector(self, product_vectors: Dict[int, np.ndarray]) -> np.ndarray:
        """
        상품 벡터들의 평균 벡터 계산
        
        Args:
            product_vectors: 상품 ID를 키로, 벡터를 값으로 하는 딕셔너리
            
        Returns:
            평균 벡터
        """
        if not product_vectors:
            return None
            
        # 모든 벡터를 합하고 개수로 나누어 평균 계산
        vectors = list(product_vectors.values())
        converted_vectors = []
        
        # 각 벡터를 숫자형 배열로 변환
        for v in vectors:
            try:
                # 문자열인 경우 처리 (PostgreSQL 벡터 문자열 형식 '[1,2,3,...]' 처리)
                if isinstance(v, str):
                    # 문자열 벡터를 정제 ('[', ']' 제거 후 쉼표로 분리)
                    if v.startswith('[') and v.endswith(']'):
                        v = v[1:-1]  # '[', ']' 제거
                    v_list = [float(x.strip()) for x in v.split(',')]
                    converted_vectors.append(np.array(v_list, dtype=np.float32))
                # NumPy 배열이거나 변환 가능한 객체인 경우
                elif hasattr(v, '__array__') or hasattr(v, '__iter__'):
                    converted_vectors.append(np.array(v, dtype=np.float32))
                else:
                    self.logger.warning(f"벡터 변환 불가: {type(v)}")
                    continue
            except Exception as e:
                self.logger.warning(f"벡터 변환 중 오류 발생: {str(e)}, 타입: {type(v)}")
                continue
        
        # 변환된 벡터가 없으면 None 반환
        if not converted_vectors:
            return None
        
        # 평균 계산
        average_vector = np.mean(converted_vectors, axis=0)
        
        # 정규화 (길이가 1이 되도록)
        norm = np.linalg.norm(average_vector)
        if norm > 0:
            average_vector = average_vector / norm
            
        return average_vector
    
    def find_similar_products(self, query_vector: np.ndarray, exclude_ids: List[int] = None) -> pd.DataFrame:
        """
        쿼리 벡터와 유사한 상품들 찾기
        
        Args:
            query_vector: 검색 기준 벡터
            exclude_ids: 결과에서 제외할 상품 ID 목록
            
        Returns:
            유사 상품 DataFrame (product_id, score 컬럼 포함)
        """
        if query_vector is None:
            return pd.DataFrame(columns=['product_id', 'score'])
         
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
            
            session = self.Session()
            try:
                # 쿼리 구성 (카테고리 필터링 적용)
                base_query = """
                    SELECT id, (image_vector <#> :query_vec) AS distance
                    FROM product
                    WHERE status = 'SALE'
                """
                
                # 카테고리 필터링 조건 추가 (설정에 따라)
                if self.use_category_filter:
                    category_filter = ""
                    if not self.include_similar_categories:
                        # 정확히 같은 카테고리만 필터링
                        # 참고: 실제로는 카테고리 정보를 가져와서 조건을 구성해야 함
                        # 여기서는 예시로만 작성
                        self.logger.info("카테고리 필터링 적용")
                
                # 제외할 ID가 있으면 조건 추가
                params = {"query_vec": vector_str, "top_k": self.top_n}
                if exclude_ids and len(exclude_ids) > 0:
                    base_query += " AND id NOT IN :exclude_ids"
                    params["exclude_ids"] = tuple(exclude_ids)
                    
                base_query += " ORDER BY distance LIMIT :top_k"
                
                # 쿼리 실행
                rows = session.execute(text(base_query), params).fetchall()
                
                # 결과를 DataFrame으로 변환
                result_df = pd.DataFrame([(row[0], 1.0 - row[1]) for row in rows], 
                                         columns=['product_id', 'score'])
                
                # 유사도 임계값 적용
                result_df = result_df[result_df['score'] >= self.similarity_threshold]
                
                return result_df
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"유사 상품 검색 중 오류 발생: {str(e)}")
            return pd.DataFrame(columns=['product_id', 'score'])
    
    def generate_recommendations(self, days: int = 30) -> pd.DataFrame:
        """
        사용자별 추천 상품 생성
        
        Args:
            days: 최근 몇 일 간의 장바구니 데이터를 사용할지 설정
            
        Returns:
            사용자별 추천 상품 DataFrame (member_id, product_id, score 컬럼 포함)
        """
        try:
            # 1. 사용자별 최근 장바구니 상품 데이터 가져오기
            cart_items_df = self.get_user_cart_items(days=days)
            if cart_items_df.empty:
                self.logger.warning(f"최근 {days}일간 장바구니 데이터가 없습니다.")
                return pd.DataFrame(columns=['member_id', 'product_id', 'score'])
                
            # 사용자별 최근 장바구니 상품 그룹화
            user_cart_items = {}
            for _, row in cart_items_df.iterrows():
                member_id = row['member_id']
                product_id = row['product_id']
                
                if member_id not in user_cart_items:
                    user_cart_items[member_id] = []
                    
                if product_id not in user_cart_items[member_id]:
                    user_cart_items[member_id].append(product_id)
            
            # 2. 추천 생성
            recommendations = []
            
            # 사용자별 처리
            for member_id, cart_products in user_cart_items.items():
                # 최소 상호작용 수보다 적은 상품을 가진 사용자는 건너뛰기
                if len(cart_products) < self.min_interactions:
                    self.logger.debug(f"사용자 {member_id}의 상호작용 수({len(cart_products)})가 최소 기준({self.min_interactions})보다 적습니다.")
                    continue
                    
                # 최근 항목만 사용 (설정된 최대 개수까지)
                cart_products = cart_products[:self.max_cart_items]
                
                # 상품 벡터 가져오기
                product_vectors = self.get_product_vectors(cart_products)
                if not product_vectors:
                    self.logger.debug(f"사용자 {member_id}의 장바구니 상품 벡터를 찾을 수 없습니다.")
                    continue
                
                # 평균 벡터 계산
                avg_vector = self.compute_average_vector(product_vectors)
                if avg_vector is None:
                    self.logger.debug(f"사용자 {member_id}의 평균 벡터 계산 실패")
                    continue
                
                # 유사 상품 검색
                similar_products = self.find_similar_products(avg_vector, exclude_ids=cart_products)
                if similar_products.empty:
                    self.logger.debug(f"사용자 {member_id}에 대한 유사 상품이 없습니다.")
                    continue
                
                # 추천 결과 추가
                for _, row in similar_products.iterrows():
                    recommendations.append({
                        'member_id': member_id,
                        'product_id': row['product_id'],
                        'score': row['score']
                    })
            
            # 3. 결과 반환
            if not recommendations:
                return pd.DataFrame(columns=['member_id', 'product_id', 'score'])
                
            recommendations_df = pd.DataFrame(recommendations)
            
            # 상위 N개만 유지
            final_recommendations = []
            for member_id, group in recommendations_df.groupby('member_id'):
                top_recs = group.sort_values('score', ascending=False).head(self.top_n)
                final_recommendations.append(top_recs)
            
            if not final_recommendations:
                return pd.DataFrame(columns=['member_id', 'product_id', 'score'])
                
            return pd.concat(final_recommendations, ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"추천 생성 중 오류 발생: {str(e)}")
            return pd.DataFrame(columns=['member_id', 'product_id', 'score'])
            
    def __del__(self):
        """소멸자: 리소스 정리"""
        self._close_vector_db() 