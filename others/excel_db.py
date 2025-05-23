"""
Excel 파일 기반의 테스트용 데이터베이스 모듈
"""

import pandas as pd
import json
from utils.logger import setup_logger

# 설정 파일 로드
with open('config/config.json', 'r') as f:
    CONFIG = json.load(f)

class ExcelRecommendationDB:
    """Excel 파일 기반의 테스트용 데이터베이스 클래스"""
    
    def __init__(self):
        """
        rating 값을 interaction_type으로 매핑:
        - impression_count >= 9 (-3) -> impression2
        - impression_count >= 6 (-2) -> impression2
        - impression_count >= 3 (-1) -> impression1
        - view_count is not null (3) -> view1
        - view_count >= 3 (3.8) -> view2
        - like_count is not null (4) -> like
        - cart_count is not null (4.5) -> cart
        - purchase_count <> 0 (5) -> purchase
        """
        self.rating_mapping = {
            "-3.0": "impression2",  # 9회 이상 노출
            "-2.0": "impression2",  # 6회 이상 노출
            "-1.0": "impression1",  # 3회 이상 노출
            "3.0": "view1",        # 일반 조회
            "3.8": "view2",      # 3회 이상 조회
            "4.0": "like",         # 좋아요
            "4.5": "cart",       # 장바구니
            "5.0": "purchase"      # 구매
        }
        # 로거 설정
        self.logger = setup_logger('excel_db')
    
    def get_user_item_interactions(self, days: int = 30) -> pd.DataFrame:
        """테스트용 상호작용 데이터 로드
        
        Args:
            days (int): 사용하지 않음 (실제 DB와의 인터페이스 통일을 위해 유지)
            
        Returns:
            pd.DataFrame: 변환된 상호작용 데이터
        """
        try:
            # Input 데이터 로드
            input_df = pd.read_csv('database/excel/input.csv')
            
            # rating 값을 interaction_type으로 매핑
            input_df['interaction_type'] = input_df['rating'].map(lambda x: self.rating_mapping[str(float(x))])
            
            # 필요한 컬럼만 선택
            result_df = input_df[['member_id', 'product_id', 'interaction_type']]
            
            # 데이터 통계 로깅
            self.logger.info(f"총 {len(result_df)}개의 상호작용 데이터 로드 완료")
            self.logger.info(f"고유 사용자 수: {result_df['member_id'].nunique()}")
            self.logger.info(f"고유 상품 수: {result_df['product_id'].nunique()}")
            
            # 상호작용 타입별 통계
            interaction_stats = result_df.groupby('interaction_type').size()
            self.logger.info("\n상호작용 타입별 통계:")
            for interaction_type, count in interaction_stats.items():
                self.logger.info(f"- {interaction_type}: {count}건")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Excel 데이터 로드 중 오류 발생: {str(e)}")
            raise 