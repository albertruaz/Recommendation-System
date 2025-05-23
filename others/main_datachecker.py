"""
ALS 기반 추천 시스템 실행 스크립트
"""

import os
import json
import pandas as pd
from database.db import RecommendationDB
from database.excel_db import ExcelRecommendationDB
from models.pyspark_als import PySparkALS
from utils.logger import setup_logger

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

# 로거 설정
logger = setup_logger('main')

def load_interactions(days: int = 30, use_test_db: bool = False) -> pd.DataFrame:
    """데이터베이스에서 상호작용 데이터 로드"""
    logger.info(f"days: {days}")
    
    if use_test_db:
        db = ExcelRecommendationDB()
    else:
        db = RecommendationDB()
        
    interactions = db.get_user_item_interactions(days=days)
    
    if interactions.empty:
        logger.error(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
        raise ValueError(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
    
    logger.info(f"총 {len(interactions)}개의 상호작용 데이터 로드 완료")
    return interactions

def main():
    # 설정 로드
    logger.info(f"Start ALS")
    days = CONFIG['default_params']['days']
    top_n = CONFIG['default_params']['top_n']
    output_dir = CONFIG['default_params']['output_dir']
    
    try:
        # 데이터 로드 (테스트 DB 사용)
        interactions_df = load_interactions(days=days, use_test_db=True)

        # 모델 타입 확인 및 모델 초기화
        model_type = CONFIG['model_type']
        logger.info(f"사용할 모델: {model_type}")
        
        model = PySparkALS(
            max_iter=CONFIG['pyspark_als']['max_iter'],
            reg_param=CONFIG['pyspark_als']['reg_param'],
            rank=CONFIG['pyspark_als']['rank'],
            interaction_weights=CONFIG['pyspark_als']['interaction_weights'],
            max_prediction=50.0,  # 예측값 상한 설정
            huber_delta=10.0      # Huber Loss의 델타값 설정
        )
        
        # 행렬 데이터 준비
        logger.info("상호작용 데이터를 행렬로 변환하는 중...")
        
        # 모든 모델 타입에 대해 공통적으로 prepare_matrices 메서드 호출
        matrix_data = model.prepare_matrices(interactions_df)
        
        logger.info("행렬 변환 완료")
        
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise
    finally:
        if 'model' in locals():
            model.cleanup()

if __name__ == "__main__":
    main() 