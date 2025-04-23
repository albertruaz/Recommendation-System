"""
ALS 기반 추천 시스템 실행 스크립트
"""

import os
import json
import pandas as pd
from database.recommendation_db import RecommendationDB
from database.excel_db import ExcelRecommendationDB

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
        
        # 모델 객체 초기화
        if model_type == 'custom_implicit_als':
            # CustomImplicitALS 모델 사용
            from models.custom_implicit_als import CustomImplicitALS
            model = CustomImplicitALS(
                max_iter=CONFIG['custom_implicit_als']['max_iter'],
                reg_param=CONFIG['custom_implicit_als']['reg_param'],
                rank=CONFIG['custom_implicit_als']['rank']
            )
        elif model_type == 'implicit_als':
            # ImplicitALS 모델 사용
            from models.implicit_als import ImplicitALS
            model = ImplicitALS(
                max_iter=CONFIG['implicit_als']['max_iter'],
                reg_param=CONFIG['implicit_als']['reg_param'],
                rank=CONFIG['implicit_als']['rank'],
                alpha=CONFIG['implicit_als']['alpha']
            )
        elif model_type == 'pyspark_als':
            # PySparkALS 모델 사용
            from models.pyspark_als import PySparkALS
            model = PySparkALS(
                max_iter=CONFIG['pyspark_als']['max_iter'],
                reg_param=CONFIG['pyspark_als']['reg_param'],
                rank=CONFIG['pyspark_als']['rank']
            )
        else:
            # 기본값: Buffalo ALS 모델 사용 (명시적 피드백)
            from models.buffalo_als import BuffaloALS
            model = BuffaloALS(
                max_iter=CONFIG['buffalo_als']['max_iter'],
                reg_param=CONFIG['buffalo_als']['reg_param'],
                rank=CONFIG['buffalo_als']['rank'],
                alpha=CONFIG['buffalo_als']['alpha']
            )
        
        # 행렬 데이터 준비
        logger.info("상호작용 데이터를 행렬로 변환하는 중...")
        
        # 모든 모델 타입에 대해 공통적으로 prepare_matrices 메서드 호출
        matrix_data = model.prepare_matrices(interactions_df)
        
        logger.info("행렬 변환 완료")
        
        # 모델 학습 - 준비된 행렬 데이터와 원본 상호작용 데이터 모두 전달
        logger.info("모델 학습 시작")
        model.train(interactions_df=interactions_df, matrix_data=matrix_data)
        
        # 추천 생성
        recommendations_df = model.generate_recommendations(top_n=top_n)
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'recommendations_{days}days.csv')
        recommendations_df.to_csv(output_path, index=False)
        logger.info(f"추천 결과가 {output_path}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise
    finally:
        if 'model' in locals():
            model.cleanup()

if __name__ == "__main__":
    main() 