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
        recommendations_df (pd.DataFrame 또는 dict): 추천 결과 데이터프레임 또는 딕셔너리
        output_dir (str): 출력 디렉토리 경로
        file_name (str, optional): 파일 이름 (없으면 기본 이름 사용)
    """
    # 딕셔너리인 경우 DataFrame으로 변환
    if isinstance(recommendations_df, dict):
        # 중첩된 딕셔너리인지 확인
        is_nested = any(isinstance(v, dict) for v in recommendations_df.values())
        
        if is_nested:
            # 중첩된 딕셔너리는 정규화하여 DataFrame으로 변환
            rows = []
            for k, v in recommendations_df.items():
                if isinstance(v, dict):
                    row = {'metric': k, **v}
                    rows.append(row)
                else:
                    rows.append({'metric': k, 'value': v})
            recommendations_df = pd.DataFrame(rows)
        else:
            # 일반 딕셔너리는 키-값 쌍으로 변환
            recommendations_df = pd.DataFrame(list(recommendations_df.items()), columns=['metric', 'value'])
    
    # None이거나 빈 데이터프레임인 경우 처리
    if recommendations_df is None or (hasattr(recommendations_df, 'empty') and recommendations_df.empty):
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
            file_name = f'rec_{run_id}'
        else:
            # 기존 방식으로 타임스탬프 사용
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'rec_{timestamp}'
    
    # 확장자가 없으면 .csv 추가
    if not file_name.endswith('.csv'):
        file_name = f'{file_name}.csv'
    
    # 파일 경로 생성
    output_path = os.path.join(output_dir, file_name)
    
    # CSV 파일로 저장
    recommendations_df.to_csv(output_path, index=False)
    logger.info(f"추천 결과가 {output_path}에 저장되었습니다.")
    
    # 결과 요약 로깅
    if hasattr(recommendations_df, 'columns'):
        if 'member_id' in recommendations_df.columns:
            user_count = recommendations_df['member_id'].nunique()
            logger.info(f"총 {user_count}명의 사용자를 위한 추천 결과입니다.")
        
        if 'product_id' in recommendations_df.columns:
            product_count = recommendations_df['product_id'].nunique()
            logger.info(f"총 {product_count}개의 상품이 추천되었습니다.")
        
        logger.info(f"추천 결과 총 {len(recommendations_df)}개의 행이 저장되었습니다.")
    