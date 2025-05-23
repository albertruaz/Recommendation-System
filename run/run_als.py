"""
ALS 기반 추천 시스템 실행 클래스
"""

import os
import json
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from database.recommendation_db import RecommendationDB
from model_als.pyspark_als import PySparkALS
from utils.logger import setup_logger
from utils.recommendation_utils import save_recommendations

from sklearn.model_selection import train_test_split

class RunALS:
    def __init__(self, run_id=None):
        # 설정 파일 로드
        with open('config/als_config.json', 'r') as f:
            self.als_config = json.load(f)
        
        # 로거 설정
        self.logger = setup_logger('run_als')
        
        # 실행 ID 설정
        self.run_id = run_id
        
        # 기본 설정 로드
        self.days = self.als_config['default_params']['days']
        self.top_n = self.als_config['default_params']['top_n']
        self.output_dir = self.als_config['default_params']['output_dir']
        
        # 테스트 설정 로드
        self.split_test_data = self.als_config['testing'].get('split_test_data', False)
        self.test_ratio = self.als_config['testing'].get('test_ratio', 0.2)
        self.random_seed = self.als_config['testing'].get('random_seed', 42)
        self.enable_loss_calculation = self.als_config['testing'].get('calculate_loss', True)
        
        # 모델 매개변수 로드
        model_params = self.als_config['pyspark_als']
        self.max_iter = model_params['max_iter']
        self.reg_param = model_params['reg_param']
        self.rank = model_params['rank']
        self.interaction_weights = model_params['interaction_weights']
        self.nonnegative = model_params.get('nonnegative', True)
        self.cold_start_strategy = model_params.get('cold_start_strategy', 'nan')
        
        self.model = None
        self.test_data = None

    def load_interactions(self) -> pd.DataFrame:
        """상호작용 데이터 로드"""
        db = RecommendationDB()  
        interactions = db.get_user_item_interactions(days=self.days, use_cache=True)
        
        if interactions.empty:
            self.logger.error(f"최근 {self.days}일 간의 상호작용 데이터가 없습니다.")
            raise ValueError(f"최근 {self.days}일 간의 상호작용 데이터가 없습니다.")
        self.logger.info(f"총 {len(interactions)}개의 상호작용 데이터 로드 완료")

        return interactions
    
    def split_data(self, interactions_df):
        """데이터 학습/테스트 분할"""
        if not self.split_test_data:
            return interactions_df, None
        
        train_df, test_df = train_test_split(
            interactions_df, 
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=interactions_df['interaction_type'] if 'interaction_type' in interactions_df.columns else None
        )
        
        return train_df, test_df
    
    def calculate_loss(self, data_df, data_type='test'):
        """데이터에 대한 손실 계산 및 예측 결과 생성"""
        if data_df is None or data_df.empty or self.model is None:
            return None
        try:
            # 모델의 test 메서드를 사용하여 데이터에 대한 예측 생성
            result_df = self.model.test(data_df)
            
            # 유효한 예측이 있는 행만 필터링
            valid_predictions = result_df.dropna(subset=['prediction'])
            
            if len(valid_predictions) == 0:
                return {
                    "mae": float('nan'), 
                    "rmse": float('nan'), 
                    "samples": 0
                }
            
            # 손실 계산
            if 'rating' not in valid_predictions.columns and 'interaction_type' in valid_predictions.columns:
                # 인스턴스 변수에서 interaction_weights 직접 사용
                valid_predictions['rating'] = valid_predictions['interaction_type'].map(self.interaction_weights)
            
            actual = valid_predictions['rating'].values
            pred = valid_predictions['prediction'].values
            
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean(np.square(actual - pred)))
            
            result = {
                "mae": float(mae),
                "rmse": float(rmse),
                "samples": len(valid_predictions)
            }
            return result_df, result
            
        except Exception as e:
            self.logger.error(f"{data_type.capitalize()} 손실 계산 중 오류 발생: {str(e)}")
            return None
    
    def run(self):
        """ALS 모델 실행 및 추천 생성"""
        self.logger.info(f"Start ALS")
        
        try:
            # 실행 ID 사용 (없는 경우 자동 생성)
            if self.run_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]  # UUID의 첫 8자만 사용
                self.run_id = f"{timestamp}_{unique_id}"
                
            self.logger.info(f"실행 ID: {self.run_id}")
            
            # 상호작용 데이터 로드
            interactions_df = self.load_interactions()
            
            # 데이터 분할 (필요한 경우)
            train_df, self.test_data = self.split_data(interactions_df)
            
            # 모델 초기화 (init에서 로드한 매개변수 사용)
            self.model = PySparkALS(
                max_iter=self.max_iter,
                reg_param=self.reg_param,
                rank=self.rank,
                interaction_weights=self.interaction_weights,
                max_prediction=50.0,
                huber_delta=10.0,
                split_test_data=self.split_test_data,
                nonnegative=self.nonnegative,
                cold_start_strategy=self.cold_start_strategy
            )
            
            # 전체 데이터로 인덱스 구성, 학습 데이터만 사용하여 학습
            prepare_df = self.model.prepare_df(interactions_df, train_df=train_df)
            
            # 모델 학습
            self.logger.info("모델 학습 시작")
            self.model.train(prepare_df=prepare_df)
            
            # 추천 생성
            recommendations_df = self.model.generate_recommendations(top_n=self.top_n)
            
            # 파일 이름에 run_id 추가
            output_dir_with_id = os.path.join(self.output_dir, self.run_id)
            os.makedirs(output_dir_with_id, exist_ok=True)
            
            user_factors_df, item_factors_df = self.model.get_latent_factors()

            train_result = None
            test_result = None
            if self.enable_loss_calculation:
                train_df, train_result = self.calculate_loss(train_df, 'train')
            if self.split_test_data and self.enable_loss_calculation and self.test_data is not None:
                test_df, test_result = self.calculate_loss(self.test_data, 'test')
                save_recommendations(test_df, output_dir=output_dir_with_id, file_name='als_4_test_result')
            
            
            save_recommendations(user_factors_df, output_dir=output_dir_with_id, file_name='als_1_user_factors')
            save_recommendations(item_factors_df, output_dir=output_dir_with_id, file_name='als_2_item_factors')
            save_recommendations(train_df, output_dir=output_dir_with_id, file_name='als_3_train_result')
            save_recommendations(recommendations_df, output_dir=output_dir_with_id, file_name='als_5_recommendations')
            
            result = {
                "run_id": self.run_id,
                "recommendations": recommendations_df,
                "train_result": train_result,
                "test_result": test_result,
                "output_dir": output_dir_with_id,
                "config": self.als_config
            }
            return result
            
        except Exception as e:
            self.logger.error(f"오류 발생: {str(e)}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if self.model is not None:
            self.model.cleanup()
            self.model = None
        self.test_data = None

# 직접 실행할 경우의 예시
if __name__ == "__main__":
    als_runner = RunALS()
    result = als_runner.run()
    if 'recommendations' in result:
        print(f"추천 결과: {len(result['recommendations'])}개")
    if 'train_result' in result and result['train_result'] is not None:
        print(f"훈련 데이터 MAE: {result['train_result']['mae']:.4f}")
        print(f"훈련 데이터 RMSE: {result['train_result']['rmse']:.4f}")
    if 'test_result' in result and result['test_result'] is not None:
        print(f"테스트 데이터 MAE: {result['test_result']['mae']:.4f}")
        print(f"테스트 데이터 RMSE: {result['test_result']['rmse']:.4f}") 