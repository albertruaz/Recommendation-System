"""
ALS 기반 추천 시스템 실행 스크립트

이 스크립트는 Implicit ALS를 사용하여 전체 사용자-상품 추천을 생성합니다.
"""

import os
import argparse
import logging
import pandas as pd

from utils import setup_logging
from database.recommendation_db import RecommendationDB
from models.als import ALSRecommender

class RecommendationError(Exception):
    """추천 시스템 관련 예외"""
    pass

def get_data_from_db(days: int = 30) -> pd.DataFrame:
    """
    데이터베이스에서 사용자-상품 상호작용 데이터를 가져옵니다.
    
    Args:
        days (int): 최근 몇 일간의 데이터를 가져올지 지정
        
    Returns:
        pd.DataFrame: 상호작용 데이터
    """
    try:
        db = RecommendationDB()
        interactions = db.get_user_item_interactions(days=days)
        
        if interactions.empty:
            raise RecommendationError(f"최근 {days}일 간의 상호작용 데이터가 없습니다.")
        
        return interactions
    
    except Exception as e:
        logging.error(f"데이터베이스에서 데이터 가져오기 실패: {str(e)}")
        raise RecommendationError(f"데이터베이스 오류: {str(e)}")

def generate_recommendations(interactions_df: pd.DataFrame, top_n: int = 300) -> pd.DataFrame:
    """
    Implicit ALS를 사용하여 전체 사용자에 대한 추천을 생성합니다.
    """
    recommender = None
    try:
        # ALS 추천 시스템 초기화
        recommender = ALSRecommender(
            max_iter=15,
            reg_param=0.1,
            rank=10,
            cold_start_strategy="drop"
        )
        
        # 모델 학습
        rmse = recommender.train(interactions_df)
        print(f"Root-mean-square error (RMSE): {rmse:.2f}")
        
        # 추천 생성
        recommendations_df = recommender.generate_recommendations(top_n)
        return recommendations_df
        
    except Exception as e:
        logging.error(f"Implicit ALS 추천 생성 실패: {str(e)}")
        raise Exception(f"Implicit ALS 오류: {str(e)}")
    
    finally:
        if recommender is not None:
            recommender.cleanup()

def main():
    parser = argparse.ArgumentParser(description='ALS 기반 추천 시스템')
    parser.add_argument('--days', type=int, default=30,
                      help='최근 몇 일간의 데이터를 사용할지 지정 (기본값: 30)')
    parser.add_argument('--top_n', type=int, default=300,
                      help='각 사용자에게 추천할 상품 수 (기본값: 300)')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='결과를 저장할 디렉토리 (기본값: output)')
    parser.add_argument('--verbose', action='store_true',
                      help='상세 로깅 활성화')
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 데이터베이스에서 상호작용 데이터 가져오기
        print("\n=== 데이터베이스에서 상호작용 데이터 가져오기 ===")
        interactions_df = get_data_from_db(days=args.days)
        print(f"총 상호작용 수: {len(interactions_df)}")
        print(f"고유 사용자 수: {interactions_df['member_id'].nunique()}")
        print(f"고유 상품 수: {interactions_df['product_id'].nunique()}")
        
        # 추천 생성
        print("\n=== Implicit ALS 모델로 추천 생성 시작 ===")
        recommendations_df = generate_recommendations(
            interactions_df=interactions_df,
            top_n=args.top_n
        )
        
        # 결과 출력 (처음 5개만)
        print("\n=== 추천 결과 샘플 (처음 5개) ===")
        print(recommendations_df.head())
        print(f"총 추천 수: {len(recommendations_df)}")
        
        # 결과를 CSV 파일로 저장
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'recommendations_{args.days}days.csv')
        recommendations_df.to_csv(output_path, index=False)
        print(f"\n추천 결과가 {output_path}에 저장되었습니다.")
    
    except RecommendationError as e:
        print(f"추천 생성 실패: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        logging.exception("상세 오류 정보:")
        exit(1)

if __name__ == "__main__":
    main() 