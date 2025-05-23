"""
장바구니 상품 유사도 기반 추천 시스템 실행 클래스
모델 기능과 데이터 가져오기 기능을 통합하여 구현
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from database.db import db
from database.vector_db import vector_db
from utils.logger import setup_logger
from utils.recommendation_utils import save_recommendations

class RunSimilars:
    def __init__(self, run_id=None):
        """
        장바구니 상품 유사도 기반 추천 실행 클래스 초기화
        
        Args:
            run_id (str, optional): 실행 ID. 없으면 자동 생성
        """
        # 설정 파일 로드
        with open('config/similars_config.json', 'r') as f:
            self.similars_config = json.load(f)
        
        # 로거 설정
        self.logger = setup_logger('similars')
        
        # 실행 ID 설정
        self.run_id = run_id
        
        # 기본 설정 로드
        self.days = self.similars_config['default_params']['days']
        self.top_n = self.similars_config['default_params']['top_n']
        self.output_dir = self.similars_config['default_params']['output_dir']
        self.verbose = self.similars_config['default_params'].get('verbose', True)
        
        # 모델 파라미터 로드
        self.model_params = self.similars_config['similars_model']
        self.similarity_threshold = self.model_params.get('similarity_threshold', 0.3)
        self.min_interactions = self.model_params.get('min_interactions', 2)
        self.max_cart_items = self.model_params.get('max_cart_items', 20)
        self.use_category_filter = self.model_params.get('use_category_filter', True)
        self.include_similar_categories = self.model_params.get('include_similar_categories', False)
        self.rec_by_style = self.model_params.get('rec_by_style', False)
        self.rec_by_each = self.model_params.get('rec_by_each', False)
        self.top_n_for_groups = self.model_params.get('top_n_for_groups', 50)
        self.top_n_for_each = self.model_params.get('top_n_for_each', 100)
        
        self.output_dir_with_id = None
        # DB 연결 초기화
        self.db = db()
        self.vector_db = vector_db()
        
    
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
    
    def generate_user_recommendations(self, days: int = 30) -> pd.DataFrame:
        """
        사용자별 추천 상품 생성
        
        Args:
            days: 최근 몇 일 간의 장바구니 데이터를 사용할지 설정
            
        Returns:
            사용자별 추천 상품 DataFrame (member_id, product_id, score, rec_type 컬럼 포함)
        """
        try:
            # 1. 사용자별 최근 장바구니 상품 데이터 가져오기 (최소 상호작용 수 조건 포함, 최대 상품 수 제한 포함)
            cart_items_df = self.db.get_recent_cart_items(
                days=days, 
                min_interactions=self.min_interactions,
                max_cart_items=self.max_cart_items
            )
            if cart_items_df.empty:
                self.logger.warning(f"최근 {days}일간 구매 데이터가 없거나 조건을 만족하는 사용자가 없습니다.")
                return pd.DataFrame(columns=['member_id', 'product_id', 'score', 'rec_type'])

            user_cart_df = (
                cart_items_df
                .groupby(['member_id', 'product_id', 'primary_category_id', 'secondary_category_id'])['styles_id']
                .agg(lambda x: list(dict.fromkeys(x)))  # 중복 제거 및 순서 유지
                .reset_index()
            )
            
            # 저장 객체 이름에 설정 정보를 포함
            save_recommendations(user_cart_df, output_dir=self.output_dir_with_id, file_name="similars_1_cart_items")

            # 2. 추천 생성
            all_recommendations = []
            
            for member_id, group in user_cart_df.groupby('member_id'):
                cart_products = group['product_id'].unique().tolist()
                
                # 스타일별 추천 생성
                if self.rec_by_style:
                    style_recommendations = self._generate_style_recommendations(member_id, group, cart_products)
                    all_recommendations.extend(style_recommendations)
                
                # 사용자별 추천 생성
                if self.rec_by_each:
                    each_recommendations = self._generate_each_recommendations(member_id, group, cart_products)
                    all_recommendations.extend(each_recommendations)
            
            if not all_recommendations:
                return pd.DataFrame(columns=['member_id', 'product_id', 'score', 'rec_type'])
                
            recommendations_df = pd.DataFrame(all_recommendations)

            # 3. 최종 추천 결과 생성 (각 타입별로 개수 제한)
            final_recommendations = []
            
            for member_id, user_group in recommendations_df.groupby('member_id'):
                for rec_type, type_group in user_group.groupby('rec_type'):
                    if rec_type == 'style':
                        top_recs = type_group.sort_values('score', ascending=False).head(self.top_n_for_groups)
                    else:  # rec_type == 'each'
                        top_recs = type_group.sort_values('score', ascending=False).head(self.top_n_for_each)
                    final_recommendations.append(top_recs)
            
            if final_recommendations:
                return pd.concat(final_recommendations, ignore_index=True)
            else:
                return pd.DataFrame(columns=['member_id', 'product_id', 'score', 'rec_type'])
            
        except Exception as e:
            self.logger.error(f"추천 생성 중 오류 발생: {str(e)}")
            return pd.DataFrame(columns=['member_id', 'product_id', 'score', 'rec_type'])
    
    def _generate_style_recommendations(self, member_id: int, group: pd.DataFrame, cart_products: List[int]) -> List[Dict]:
        """스타일별 추천 생성"""
        recommendations = []
        
        # 스타일별로 그룹화하여 처리
        style_groups = {}
        for _, row in group.iterrows():
            product_id = row['product_id']
            styles_id = row['styles_id']
            # styles_id를 집합으로 변환 (중복 제거 및 순서 무관하게)
            if isinstance(styles_id, list):
                style_key = tuple(sorted(set(styles_id)))  # 집합을 정렬된 튜플로 변환
            else:
                style_key = (styles_id,) if styles_id is not None else ()
            
            if style_key not in style_groups:
                style_groups[style_key] = []
            style_groups[style_key].append(product_id)
        
        # 각 스타일 그룹별로 추천 생성
        for style_key, style_products in style_groups.items():
            product_vectors = self.vector_db.get_product_vectors(style_products)
            if not product_vectors:
                self.logger.debug(f"사용자 {member_id}의 스타일 {style_key} 상품 벡터를 찾을 수 없습니다.")
                continue

            avg_vector = self.compute_average_vector(product_vectors)
            if avg_vector is None:
                self.logger.debug(f"사용자 {member_id}의 스타일 {style_key} 평균 벡터 계산 실패")
                continue
                
            similar_products = self.vector_db.search_by_vector(avg_vector, top_k=self.top_n_for_groups, exclude_ids=cart_products)
            if not similar_products:
                self.logger.debug(f"사용자 {member_id}의 스타일 {style_key}에 대한 유사 상품이 없습니다.")
                continue
            
            for product_id, distance in similar_products:
                score = 1.0 - distance  # 거리를 유사도 점수로 변환
                if score >= self.similarity_threshold:
                    recommendations.append({
                        'member_id': member_id,
                        'product_id': product_id,
                        'score': score,
                        'rec_type': 'style'
                    })
        
        return recommendations
    
    def _generate_each_recommendations(self, member_id: int, group: pd.DataFrame, cart_products: List[int]) -> List[Dict]:
        """사용자별 추천 생성"""
        recommendations = []
        
        # 기존 방식: 사용자별로 모든 상품의 평균 벡터 계산
        product_vectors = self.vector_db.get_product_vectors(cart_products)
        if not product_vectors:
            self.logger.debug(f"사용자 {member_id}의 장바구니 상품 벡터를 찾을 수 없습니다.")
            return recommendations

        avg_vector = self.compute_average_vector(product_vectors)
        if avg_vector is None:
            self.logger.debug(f"사용자 {member_id}의 평균 벡터 계산 실패")
            return recommendations
            
        similar_products = self.vector_db.search_by_vector(avg_vector, top_k=self.top_n_for_each, exclude_ids=cart_products)
        if not similar_products:
            self.logger.debug(f"사용자 {member_id}에 대한 유사 상품이 없습니다.")
            return recommendations
        
        for product_id, distance in similar_products:
            score = 1.0 - distance  # 거리를 유사도 점수로 변환
            if score >= self.similarity_threshold:
                recommendations.append({
                    'member_id': member_id,
                    'product_id': product_id,
                    'score': score,
                    'rec_type': 'each'
                })
        
        return recommendations
    
    def run(self):
        """
        장바구니 상품 유사도 기반 추천 생성 실행
        
        Returns:
            추천 결과 정보를 담고 있는 딕셔너리
        """
        self.logger.info("장바구니 상품 유사도 기반 추천 시작")
        
        try:
            # 실행 ID 사용 (없는 경우 자동 생성)
            if self.run_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.run_id = f"similars_{timestamp}"
                
            self.logger.info(f"실행 ID: {self.run_id}")
            
            # 출력 디렉토리 생성
            output_dir_with_id = os.path.join(self.output_dir, self.run_id)
            os.makedirs(output_dir_with_id, exist_ok=True)
            self.output_dir_with_id = output_dir_with_id
            
            # 활성화된 추천 타입 로그
            active_types = []
            if self.rec_by_style:
                active_types.append("스타일별")
            if self.rec_by_each:
                active_types.append("사용자별")
            self.logger.info(f"활성화된 추천 타입: {', '.join(active_types) if active_types else '없음'}")
            
            # 추천 생성 - 이제 모델 대신 직접 RunSimilars에서 구현된 메서드 호출
            self.logger.info(f"최근 {self.days}일간의 장바구니 데이터로 추천 생성 중...")
            recommendations_df = self.generate_user_recommendations(days=self.days)
            
            save_recommendations(recommendations_df, output_dir=output_dir_with_id, file_name="similars_1_recommendations")

            # 추천 타입별 통계 계산
            type_stats = {}
            if not recommendations_df.empty:
                for rec_type in recommendations_df['rec_type'].unique():
                    type_df = recommendations_df[recommendations_df['rec_type'] == rec_type]
                    type_stats[rec_type] = {
                        'count': len(type_df),
                        'users': type_df['member_id'].nunique(),
                        'products': type_df['product_id'].nunique()
                    }

            # 결과 반환 (run_id와 config 추가)
            result = {
                "run_id": self.run_id,
                "recommendations": recommendations_df,
                "user_count": recommendations_df['member_id'].nunique() if not recommendations_df.empty else 0,
                "product_count": recommendations_df['product_id'].nunique() if not recommendations_df.empty else 0,
                "total_recommendations": len(recommendations_df),
                "type_stats": type_stats,
                "output_dir": output_dir_with_id,
                "config": self.similars_config
            }
            
            self.logger.info("장바구니 상품 유사도 기반 추천 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"추천 생성 중 오류 발생: {str(e)}", exc_info=True)
            raise
        finally:
            # DB 연결 닫기
            if hasattr(self, 'vector_db'):
                self.vector_db.close()

# 직접 실행할 경우의 예시
if __name__ == "__main__":
    try:
        runner = RunSimilars()
        result = runner.run()
        
        if 'recommendations' in result and not result['recommendations'].empty:
            print(f"추천 결과: {result['total_recommendations']}개")
            print(f"추천 대상 사용자 수: {result['user_count']}명")
            print(f"추천된 상품 수: {result['product_count']}개")
            
            # 타입별 통계 출력
            if 'type_stats' in result and result['type_stats']:
                print("\n=== 추천 타입별 통계 ===")
                for rec_type, stats in result['type_stats'].items():
                    type_name = "스타일별" if rec_type == "style" else "사용자별"
                    print(f"{type_name} 추천: {stats['count']}개 (사용자 {stats['users']}명, 상품 {stats['products']}개)")
            
            print(f"결과 저장 경로: {result['output_dir']}")
        else:
            print("추천 결과가 없습니다.")
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}") 