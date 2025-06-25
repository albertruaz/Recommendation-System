import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING, List, Set
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from core.data_loader import DataLoader

from core.model import ALSModel
from utils.logger import setup_logger


class RecommendationService:
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('recommendation_service')
        
        model_params = {
            'max_iter': config['max_iter'],
            'reg_param': config['reg_param'],
            'rank': config['rank'],
            'nonnegative': config['nonnegative'],
            'cold_start_strategy': config['cold_start_strategy']
        }
        self.model = ALSModel(**model_params)
    
    def run_recommendation(self, ratings_df: pd.DataFrame) -> Dict[int, List[int]]:
        try:
            train_df, test_df = self._split_data(ratings_df)
            self.model.train(train_df)
            recommendations = self._generate_recommendations(top_n=self.config['top_n'])
            
            if test_df is not None:
                self._log_evaluation_results(test_df)

            return recommendations
            
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
        
        self.logger.info("데이터 분할")
        self.logger.info(f"학습 데이터: {len(train_df):,}개")
        self.logger.info(f"테스트 데이터: {len(test_df):,}개")
        
        return train_df, test_df
    
    def _generate_recommendations(self, top_n: int) -> Dict[int, List[int]]:
        self.logger.info("추천 생성")
        
        recommendations = self.model.recommend_for_all_users(top_n)
        
        self.logger.info(f"생성된 추천: {len(recommendations):,}명의 사용자")
        
        return recommendations
    
    def _log_evaluation_results(self, test_df: pd.DataFrame) -> None:
        """평가 결과를 로깅합니다."""
        try:
            predictions_df = self.model.predict(test_df)
            valid_predictions = predictions_df.dropna(subset=['prediction'])
            
            if len(valid_predictions) == 0:
                self.logger.warning("유효한 예측 결과가 없습니다.")
                return
            
            actual = valid_predictions['rating'].values
            predicted = valid_predictions['prediction'].values
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean(np.square(actual - predicted)))
            
            self.logger.info("모델 평가 결과")
            self.logger.info(f"평가 샘플 수: {len(valid_predictions):,}개")
            self.logger.info(f"MAE: {mae:.4f}")
            self.logger.info(f"RMSE: {rmse:.4f}")
            
        except Exception as e:
            self.logger.error(f"모델 평가 중 오류: {str(e)}")
    
