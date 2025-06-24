import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from core.data_loader import DataLoader

from core.model import ALSModel
from utils.logger import setup_logger


class RecommendationService:
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('recommendation_service')
        
        self.model = ALSModel(**config['model_params'])
        
        self.data_loader = None
    
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
        self.logger.info(f"최적화된 추천 생성 시작 (top_n={top_n})")
        
        user_matrix, item_matrix, user_ids, item_ids = self.model.get_factors_optimized()
        
        self.logger.info(f"매트릭스 크기: 사용자 {user_matrix.shape}, 아이템 {item_matrix.shape}")
        
        all_scores = user_matrix @ item_matrix.T
        
        top_items_indices = np.argpartition(all_scores, -top_n, axis=1)[:, -top_n:]
        
        for user_idx in range(len(top_items_indices)):
            user_top_items = top_items_indices[user_idx]
            user_scores = all_scores[user_idx, user_top_items]
            sorted_order = np.argsort(user_scores)[::-1]
            top_items_indices[user_idx] = user_top_items[sorted_order]
        
        recommendations = []
        for user_idx, top_items in enumerate(top_items_indices):
            user_id = self.data_loader.idx2user[user_ids[user_idx]]
            for item_idx in top_items:
                item_id = self.data_loader.idx2item[item_ids[item_idx]]
                score = all_scores[user_idx, item_idx]
                recommendations.append({
                    'member_id': user_id,
                    'product_id': item_id,
                    'predicted_rating': float(score)
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        self.logger.info(f"최적화된 추천 생성 완료: {len(recommendations_df)}개")
        
        return recommendations_df
    
    def _evaluate_model(self, test_df: pd.DataFrame) -> Dict:
        if test_df is None or test_df.empty:
            return {}
        
        try:
            predictions_df = self.model.predict(test_df)
            
            valid_predictions = predictions_df.dropna(subset=['prediction'])
            
            if len(valid_predictions) == 0:
                return {'mae': float('nan'), 'rmse': float('nan'), 'samples': 0}
            
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
    
