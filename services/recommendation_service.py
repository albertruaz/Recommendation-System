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
        """추천 생성"""
        self.logger.info(f"추천 생성 시작 (top_n={top_n})")
        
        # 사용자-아이템 잠재 요인 가져오기
        user_factors, item_factors = self.model.get_factors()
        
        recommendations = []
        
        # 각 사용자에 대해 추천 생성
        for _, user_row in user_factors.iterrows():
            user_idx = user_row['id']
            user_id = self.data_loader.idx2user[user_idx]
            user_vector = np.array(user_row['features'])
            
            # 모든 아이템에 대한 점수 계산
            scores = []
            for _, item_row in item_factors.iterrows():
                item_idx = item_row['id']
                item_id = self.data_loader.idx2item[item_idx]
                item_vector = np.array(item_row['features'])
                
                score = np.dot(user_vector, item_vector)
                scores.append({
                    'member_id': user_id,
                    'product_id': item_id,
                    'predicted_rating': score
                })
            
            # 상위 N개 선택
            user_scores = sorted(scores, key=lambda x: x['predicted_rating'], reverse=True)[:top_n]
            recommendations.extend(user_scores)
        
        recommendations_df = pd.DataFrame(recommendations)
        self.logger.info(f"추천 생성 완료: {len(recommendations_df)}개")
        
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
    
