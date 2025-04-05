"""
ALS 기반 추천 시스템 실행 스크립트
"""

import os
import json
import pandas as pd
from database.recommendation_db import RecommendationDB
from models.custom_implicit_als import CustomImplicitALS
# from models.implicit_als import ImplicitALS
from utils.logger import setup_logger

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

# 로거 설정
logger = setup_logger('main')

def load_interactions(days: int = 30) -> pd.DataFrame:
    """데이터베이스에서 상호작용 데이터 로드"""
    db = RecommendationDB()
    interactions = db.get_user_item_interactions(days=days)
    
    if interactions.empty:
        logger.error(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
        raise ValueError(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
    
    logger.info(f"총 {len(interactions)}개의 상호작용 데이터 로드 완료")
    return interactions

def main():
    # 설정 로드
    days = CONFIG['default_params']['days']
    top_n = CONFIG['default_params']['top_n']
    output_dir = CONFIG['default_params']['output_dir']
    
    try:
        # 데이터 로드
        interactions_df = load_interactions(days=days)
        
        # 모델 초기화 및 학습
        # model = CustomImplicitALS(
        #     max_iter=CONFIG['als_params']['max_iter'],
        #     reg_param=CONFIG['als_params']['reg_param'],
        #     rank=CONFIG['als_params']['rank']
        # )

        # model = ImplicitALS(
        #     max_iter=CONFIG['als_params']['max_iter'],
        #     reg_param=CONFIG['als_params']['reg_param'],
        #     rank=CONFIG['als_params']['rank'],
        #     alpha=CONFIG['alpha']
        # )
        
        # 모델 학습
        model.train(interactions_df)
        
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