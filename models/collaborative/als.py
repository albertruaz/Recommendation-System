import os
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse import coo_matrix
import joblib

# 기반 클래스 임포트
from models.base.base_recommender import BaseRecommender

try:
    import implicit
except ImportError:
    raise ImportError("Implicit 라이브러리가 설치되어 있지 않습니다. 'pip install implicit' 명령어로 설치하세요.")

class ALSModel(BaseRecommender):
    """
    Implicit ALS 모델 클래스

    예시 사용:
        model = ALSModel(config_dict={
            'factors': 50,
            'regularization': 0.01,
            'iterations': 15,
            'alpha': 1.0
        })
        model.train(ratings_df)
        
        # 추천 얻기
        recs = model.get_recommendations(user_id=123, k=10)
        
        # 모든 사용자에 대한 추천 얻기
        all_recs = model.get_all_recommendations(k=10)
    """
    
    def __init__(self, config_dict=None):
        """
        Args:
            config_dict (dict, optional):
                - factors (int): 잠재 요인 차원 수
                - regularization (float): 정규화 계수
                - iterations (int): 학습 반복 횟수
                - alpha (float): 논문에서 제안된 implicit 데이터 가중치 파라미터
                - use_gpu (bool): GPU 사용 여부 (기본 False)
        """
        super().__init__()
        self.config = config_dict or {}
        
        # Implicit ALS 주요 하이퍼파라미터
        self.factors = self.config.get('factors', 10)
        self.regularization = self.config.get('regularization', 0.1)
        self.iterations = self.config.get('iterations', 15)
        self.alpha = self.config.get('alpha', 1.0)
        self.use_gpu = self.config.get('use_gpu', False)
        
        # 사용자/아이템 인덱스 맵핑 딕셔너리
        self.user2index = {}
        self.index2user = []
        self.item2index = {}
        self.index2item = []
        
        self.model = None
        self.item_user_matrix = None  # 학습에 사용된 item-user 행렬 저장
        self.is_trained = False

    def _build_mappings(self, ratings: pd.DataFrame):
        """
        사용자 ID와 아이템 ID를 0부터 시작하는 내부 인덱스로 매핑하기 위한 함수
        """
        unique_users = ratings['user_id'].unique().tolist()
        unique_items = ratings['item_id'].unique().tolist()
        
        self.user2index = {u: i for i, u in enumerate(unique_users)}
        self.index2user = unique_users
        
        self.item2index = {i: j for j, i in enumerate(unique_items)}
        self.index2item = unique_items
        

    def _create_sparse_matrix(self, ratings: pd.DataFrame):
        """
        implicit ALS가 학습할 coo_matrix 형식의 (item x user) 행렬을 만든다.
        (Implicit 공식 문서 권장: item-user 형식을 사용)
        """
        # DataFrame -> (user, item, rating)
        rows = []
        cols = []
        vals = []
        
        for row in ratings.itertuples(index=False):
            u = self.user2index[row.user_id]
            i = self.item2index[row.item_id]
            # Implicit ALS에서는 "implicit feedback" 형태를 주로 사용하므로
            # 가중치(alpha)를 반영할 수도 있고, rating을 그대로 써도 무방
            rows.append(i)      # 아이템 인덱스
            cols.append(u)      # 사용자 인덱스
            vals.append(float(row.rating) * self.alpha)
        
        # (item_count, user_count) 크기의 희소 행렬
        item_count = len(self.index2item)
        user_count = len(self.index2user)
        
        matrix = coo_matrix((vals, (rows, cols)), shape=(item_count, user_count))
        return matrix

    def train(self, ratings: pd.DataFrame) -> None:
        """
        모델 학습

        Args:
            ratings (pd.DataFrame): 'user_id', 'item_id', 'rating' 컬럼 필수
        """
        try:
            logging.info("ALS 모델 학습 시작")
            
            # 예외 처리
            if ratings.empty:
                raise ValueError("빈 데이터셋으로는 학습할 수 없습니다.")
            
            # 컬럼명 변환이 필요하면 여기서 처리
            if 'member_id' in ratings.columns and 'user_id' not in ratings.columns:
                ratings = ratings.rename(columns={'member_id': 'user_id'})
            if 'product_id' in ratings.columns and 'item_id' not in ratings.columns:
                ratings = ratings.rename(columns={'product_id': 'item_id'})
                
            required_cols = {'user_id', 'item_id', 'rating'}
            if not required_cols.issubset(ratings.columns):
                raise ValueError(f"필수 컬럼이 없습니다: {required_cols}")
            
            # (1) 유저/아이템 인덱스 매핑
            self._build_mappings(ratings)
            
            # (2) 희소 행렬 생성 (item-user)
            self.item_user_matrix = self._create_sparse_matrix(ratings)
            
            # (3) Implicit ALS 모델 초기화
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                use_gpu=self.use_gpu
            )
            
            # (4) 학습
            # 실제로는 "user_item_matrix.T"를 넣어야 하나, 우리가 만든 건 이미 item x user라 그대로 사용
            self.model.fit(self.item_user_matrix)
            
            self.is_trained = True
            logging.info("ALS 모델 학습 완료")

        except Exception as e:
            self.is_trained = False
            logging.error(f"ALS 모델 학습 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 학습 실패: {str(e)}")

    def get_user_factors(self, user_id: int) -> Optional[np.ndarray]:
        """
        특정 사용자의 잠재 요인 벡터를 반환합니다.

        Args:
            user_id (int): 사용자 ID

        Returns:
            Optional[np.ndarray]: 사용자 잠재 요인 벡터
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        if user_id not in self.user2index:
            # 학습 때 없던 사용자이면 None 반환
            return None
        
        user_idx = self.user2index[user_id]
        return self.model.user_factors[user_idx]

    def get_item_factors(self, item_id: int) -> Optional[np.ndarray]:
        """
        특정 아이템의 잠재 요인 벡터를 반환합니다.

        Args:
            item_id (int): 아이템 ID

        Returns:
            Optional[np.ndarray]: 아이템 잠재 요인 벡터
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        if item_id not in self.item2index:
            # 학습 때 없던 아이템이면 None 반환
            return None
        
        item_idx = self.item2index[item_id]
        return self.model.item_factors[item_idx]

    def get_recommendations(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        특정 사용자에 대한 추천 아이템 Top-K 추출

        Args:
            user_id (int): 사용자 ID
            k (int, optional): 추천받을 아이템 개수. 기본 10

        Returns:
            List[Tuple[int, float]]: (item_id, 추정 점수) 튜플 리스트
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        if user_id not in self.user2index:
            # 학습 때 없던 사용자이면 빈 리스트 반환 (또는 cold-start 대응 처리)
            return []
        
        try:
            # 내부 user 인덱스로 변환
            user_idx = self.user2index[user_id]
            
            # 모델의 recommend() 호출
            rec_items, rec_scores = self.model.recommend(
                userid=user_idx,
                user_items=self.item_user_matrix[:, user_idx],  # 이미 본 아이템 필터링
                N=k,
                filter_already_liked_items=True
            )
            
            # 내부 인덱스 -> 실제 item_id로 변환
            results = []
            for item_idx, score in zip(rec_items, rec_scores):
                real_item_id = self.index2item[item_idx]
                results.append((real_item_id, float(score)))
            
            return results

        except Exception as e:
            logging.error(f"추천 생성 중 오류 발생: {str(e)}")
            raise RuntimeError(f"추천 생성 실패: {str(e)}")

    def get_all_recommendations(self, k: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """
        모든 사용자에 대한 추천 아이템 Top-K 추출

        Args:
            k (int, optional): 추천받을 아이템 개수. 기본 10

        Returns:
            Dict[int, List[Tuple[int, float]]]: {user_id: [(item_id, 점수), ...]} 형태의 딕셔너리
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        try:
            all_recommendations = {}
            
            # 모든 사용자에 대해 추천 생성
            for user_id in self.index2user:
                recommendations = self.get_recommendations(user_id, k=k)
                all_recommendations[user_id] = recommendations
                
            return all_recommendations
            
        except Exception as e:
            logging.error(f"전체 추천 생성 중 오류 발생: {str(e)}")
            raise RuntimeError(f"전체 추천 생성 실패: {str(e)}")
    
    def get_recommendations_for_all_users(self, k: int = 300) -> pd.DataFrame:
        """
        모든 사용자에 대한 추천 결과를 DataFrame 형태로 반환합니다.
        
        Args:
            k (int, optional): 각 사용자당 추천할 아이템 수. 기본 300.
            
        Returns:
            pd.DataFrame: 다음 컬럼을 가진 DataFrame
                - member_id(또는 user_id): 사용자 ID
                - product_id(또는 item_id): 추천 아이템 ID
                - rating: 예측 점수
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
            
        try:
            # 모든 사용자에 대한 추천 가져오기
            all_recs = self.get_all_recommendations(k=k)
            
            # 결과를 DataFrame으로 변환
            results = []
            for user_id, recs in all_recs.items():
                for item_id, score in recs:
                    results.append({
                        'member_id': user_id,
                        'product_id': item_id,
                        'rating': score
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logging.error(f"추천 결과 변환 중 오류 발생: {str(e)}")
            raise RuntimeError(f"추천 결과 변환 실패: {str(e)}")

    def save_model(self, path: str) -> None:
        """
        학습된 모델을 파일로 저장

        Args:
            path (str): 저장할 경로
        """
        if not self.is_trained:
            raise RuntimeError("학습되지 않은 모델은 저장할 수 없습니다.")
        
        try:
            # 저장할 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 모델 컴포넌트 저장
            model_components = {
                'user2index': self.user2index,
                'index2user': self.index2user,
                'item2index': self.item2index,
                'index2item': self.index2item,
                'item_user_matrix': self.item_user_matrix,
                'model_params': {
                    'factors': self.factors,
                    'regularization': self.regularization,
                    'iterations': self.iterations,
                    'alpha': self.alpha,
                    'use_gpu': self.use_gpu,
                },
                'user_factors': self.model.user_factors if self.model else None,
                'item_factors': self.model.item_factors if self.model else None,
            }
            
            joblib.dump(model_components, path)
            logging.info(f"모델이 {path}에 저장되었습니다.")
            
        except Exception as e:
            logging.error(f"모델 저장 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 저장 실패: {str(e)}")

    def load_model(self, path: str) -> None:
        """
        저장된 모델을 로드

        Args:
            path (str): 모델 파일 경로
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
            
            # 모델 컴포넌트 로드
            model_components = joblib.load(path)
            
            # 맵핑 정보 복원
            self.user2index = model_components['user2index']
            self.index2user = model_components['index2user']
            self.item2index = model_components['item2index']
            self.index2item = model_components['index2item']
            self.item_user_matrix = model_components['item_user_matrix']
            
            # 하이퍼파라미터 설정
            params = model_components['model_params']
            self.factors = params['factors']
            self.regularization = params['regularization']
            self.iterations = params['iterations']
            self.alpha = params['alpha']
            self.use_gpu = params['use_gpu']
            
            # 모델 초기화 및 가중치 복원
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                use_gpu=self.use_gpu
            )
            
            # 학습된 가중치 복원
            if model_components['user_factors'] is not None and model_components['item_factors'] is not None:
                self.model.user_factors = model_components['user_factors']
                self.model.item_factors = model_components['item_factors']
                self.is_trained = True
                logging.info(f"모델이 {path}에서 성공적으로 로드되었습니다.")
            else:
                self.is_trained = False
                logging.warning("로드된 모델에 학습된 가중치가 없습니다.")
                
        except Exception as e:
            self.is_trained = False
            logging.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 로드 실패: {str(e)}") 