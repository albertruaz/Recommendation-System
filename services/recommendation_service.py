"""
추천 서비스 - 비즈니스 로직 담당
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from core.data_loader import DataLoader

from core.model import ALSModel
from utils.logger import setup_logger


class RecommendationService:
    """추천 시스템의 비즈니스 로직 담당"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('recommendation_service')
        
        # 서비스 초기화 (데이터 로더는 외부에서 전달받음)
        self.model = ALSModel(**config['model_params'])
        
        self.data_loader = None  # 외부에서 전달받음
    
    def run_recommendation(self, ratings_df: pd.DataFrame) -> Dict:
        try:
            train_df, test_df = self._split_data(ratings_df)
            
            self.model.train(train_df)
            
            recommendations = self._generate_recommendations(
                top_n=self.config['top_n']
            )
            
            evaluation_results = self._evaluate_model(test_df) if test_df is not None else None
            if evaluation_results:
                self.logger.info(f"평가 결과: {evaluation_results}")

            return {
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"추천 서비스 실행 중 오류: {str(e)}")
            raise
        finally:
            self.model.cleanup()
    
    def _split_data(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """데이터 분할"""
        if not self.config.get('split_test_data', False):
            return ratings_df, None
        
        train_df, test_df = train_test_split(
            ratings_df,
            test_size=self.config.get('test_ratio', 0.2),
            random_state=self.config.get('random_seed', 42)
        )
        
        self.logger.info(f"데이터 분할 완료 - Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df
    
    def _generate_recommendations(self, top_n: int) -> pd.DataFrame:
        """벡터화된 추천 생성 - O(N×M)에서 O(N+M)으로 개선"""
        self.logger.info(f"벡터화된 추천 생성 시작 (top_n={top_n})")
        
        # 사용자-아이템 잠재 요인 가져오기
        user_factors, item_factors = self.model.get_factors()
        
        # 1. 모든 벡터를 한번에 numpy 배열로 변환
        user_matrix = np.array([row['features'] for _, row in user_factors.iterrows()])  # (n_users, n_factors)
        item_matrix = np.array([row['features'] for _, row in item_factors.iterrows()])  # (n_items, n_factors)
        
        self.logger.info(f"매트릭스 크기: 사용자 {user_matrix.shape}, 아이템 {item_matrix.shape}")
        
        # 2. 매트릭스 곱셈으로 모든 점수를 한번에 계산 - 핵심 개선!
        all_scores = np.dot(user_matrix, item_matrix.T)  # (n_users, n_items)
        
        # 3. 각 사용자별로 top-N 인덱스 추출
        top_items_indices = np.argsort(all_scores, axis=1)[:, -top_n:]  # 상위 N개
        
        # 4. 결과 포맷팅
        recommendations = []
        for user_idx, top_items in enumerate(top_items_indices):
            user_id = self.data_loader.idx2user[user_factors.iloc[user_idx]['id']]
            for item_idx in reversed(top_items):  # 높은 점수부터
                item_id = self.data_loader.idx2item[item_factors.iloc[item_idx]['id']]
                score = all_scores[user_idx, item_idx]
                recommendations.append({
                    'member_id': user_id,
                    'product_id': item_id,
                    'predicted_rating': float(score)
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        self.logger.info(f"벡터화된 추천 생성 완료: {len(recommendations_df)}개")
        
        return recommendations_df
    
    def _evaluate_model(self, test_df: pd.DataFrame) -> Dict:
        """모델 평가"""
        if test_df is None or test_df.empty:
            return {}
        
        try:
            predictions_df = self.model.predict(test_df)
            
            # 유효한 예측만 필터링
            valid_predictions = predictions_df.dropna(subset=['prediction'])
            
            if len(valid_predictions) == 0:
                return {'mae': float('nan'), 'rmse': float('nan'), 'samples': 0}
            
            # MAE, RMSE 계산
            actual = valid_predictions['rating'].values
            predicted = valid_predictions['prediction'].values
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean(np.square(actual - predicted)))
            
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'samples': len(valid_predictions)
            }
            
        except Exception as e:
            self.logger.error(f"모델 평가 중 오류: {str(e)}")
            return {}
    
