"""
ALS 기반 추천 시스템 실행 클래스
"""

import os
import json
import pandas as pd
import numpy as np
from database.recommendation_db import RecommendationDB
from model_als.pyspark_als import PySparkALS
from utils.logger import setup_logger
from utils.recommendation_utils import save_recommendations

from sklearn.model_selection import train_test_split

class RunALS:
    def __init__(self):
        # 설정 파일 로드
        with open('config/als_config.json', 'r') as f:
            self.als_config = json.load(f)
        
        # 로거 설정
        self.logger = setup_logger('run_als')
        
        # 기본 설정 로드
        self.days = self.als_config['default_params']['days']
        self.top_n = self.als_config['default_params']['top_n']
        self.output_dir = self.als_config['default_params']['output_dir']
        
        # 테스트 설정 로드
        self.split_test_data = self.als_config['testing'].get('split_test_data', False)
        self.test_ratio = self.als_config['testing'].get('test_ratio', 0.2)
        self.random_seed = self.als_config['testing'].get('random_seed', 42)
        self.enable_loss_calculation = self.als_config['testing'].get('calculate_loss', True)
        
        self.model = None
        self.test_data = None

    def load_interactions(self) -> pd.DataFrame:
        """상호작용 데이터 로드"""
        db = RecommendationDB()  
        interactions = db.get_user_item_interactions(days=self.days)
        
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
                valid_predictions['rating'] = valid_predictions['interaction_type'].map(
                    self.als_config['pyspark_als']['interaction_weights']
                )
            
            actual = valid_predictions['rating'].values
            pred = valid_predictions['prediction'].values
            
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean(np.square(actual - pred)))
            
            result_path = None
            if data_type == 'test':
                os.makedirs(self.output_dir, exist_ok=True)
                result_path = os.path.join(self.output_dir, f'{data_type}_predictions_{self.days}days.csv')
                result_df.to_csv(result_path, index=False)
                self.logger.info(f"{data_type.capitalize()} 예측 결과가 {result_path}에 저장되었습니다.")
            
            result = {
                "mae": float(mae),
                "rmse": float(rmse),
                "samples": len(valid_predictions)
            }
            
            if result_path:
                result["results_path"] = result_path
                
            return result
            
        except Exception as e:
            self.logger.error(f"{data_type.capitalize()} 손실 계산 중 오류 발생: {str(e)}")
            return None
    
    def run(self):
        """ALS 모델 실행 및 추천 생성"""
        self.logger.info(f"Start ALS")
        
        try:
            # 상호작용 데이터 로드
            interactions_df = self.load_interactions()
            
            # 데이터 분할 (필요한 경우)
            train_df, self.test_data = self.split_data(interactions_df)
            
            # 모델 초기화
            self.model = PySparkALS(
                max_iter=self.als_config['pyspark_als']['max_iter'],
                reg_param=self.als_config['pyspark_als']['reg_param'],
                rank=self.als_config['pyspark_als']['rank'],
                interaction_weights=self.als_config['pyspark_als']['interaction_weights'],
                max_prediction=50.0,
                huber_delta=10.0,
                split_test_data=self.split_test_data
            )
            
            # 전체 데이터로 인덱스 구성, 학습 데이터만 사용하여 학습
            matrix_data = self.model.prepare_matrices(interactions_df, train_df=train_df)
            
            # 모델 학습
            self.logger.info("모델 학습 시작")
            self.model.train(matrix_data=matrix_data)
            
            # 추천 생성
            recommendations_df = self.model.generate_recommendations(top_n=self.top_n)
            
            train_result = None
            test_result = None
            if self.enable_loss_calculation:
                train_result = self.calculate_loss(train_df, 'train')
            if self.split_test_data and self.enable_loss_calculation and self.test_data is not None:
                test_result = self.calculate_loss(self.test_data, 'test')
            
            # 결과 반환
            result = {
                "recommendations": recommendations_df,
                "train_result": train_result,
                "test_result": test_result
            }

            if train_result is not None:
                self.logger.info(f"학습 데이터 결과:")
                self.logger.info(f"- MAE: {train_result['mae']:.4f}")
                self.logger.info(f"- RMSE: {train_result['rmse']:.4f}")
                self.logger.info(f"- 샘플 수: {train_result['samples']}")
            
            # 테스트 결과 정보 출력 (있는 경우)
            if test_result is not None:
                self.logger.info(f"테스트 데이터 결과:")
                self.logger.info(f"- MAE: {test_result['mae']:.4f}")
                self.logger.info(f"- RMSE: {test_result['rmse']:.4f}")
                self.logger.info(f"- 샘플 수: {test_result['samples']}")
            save_recommendations(recommendations_df)
            
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