"""
추천 시스템 유틸리티 함수 모듈

추천 결과 처리 및 저장과 관련된 유틸리티 함수들을 제공합니다.
"""

import os
import pandas as pd
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger('rec_utils')

def save_recommendations(recommendations_df, output_dir='output', file_name=None):
    """
    추천 결과를 CSV 파일로 저장합니다.
    
    Args:
        recommendations_df (pd.DataFrame): 추천 결과 데이터프레임
        output_dir (str): 출력 디렉토리 경로
        file_name (str, optional): 파일 이름 (없으면 기본 이름 사용)
    """
    if recommendations_df is None or recommendations_df.empty:
        logger.warning("저장할 추천 결과가 없습니다.")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 설정
    if file_name is None:
        # output_dir이 run_id를 포함하는지 확인
        dir_parts = output_dir.split('/')
        if len(dir_parts) > 0 and '_' in dir_parts[-1]:
            # 마지막 디렉토리가 run_id 형식인 경우
            run_id = dir_parts[-1]
            file_name = f'recommendations_{run_id}.csv'
        else:
            # 기존 방식으로 타임스탬프 사용
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'recommendations_{timestamp}.csv'
    
    # 파일 경로 생성
    output_path = os.path.join(output_dir, file_name)
    
    # CSV 파일로 저장
    recommendations_df.to_csv(output_path, index=False)
    logger.info(f"추천 결과가 {output_path}에 저장되었습니다.")
    
    # 결과 요약 로깅
    if 'member_id' in recommendations_df.columns:
        user_count = recommendations_df['member_id'].nunique()
        logger.info(f"총 {user_count}명의 사용자를 위한 추천 결과입니다.")
    
    if 'product_id' in recommendations_df.columns:
        product_count = recommendations_df['product_id'].nunique()
        logger.info(f"총 {product_count}개의 상품이 추천되었습니다.")
    
    logger.info(f"추천 결과 총 {len(recommendations_df)}개의 행이 저장되었습니다.")
    