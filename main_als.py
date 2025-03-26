"""
ALS 기반 추천 시스템 실행 스크립트

이 스크립트는 PySpark ALS를 사용하여 전체 사용자-상품 추천을 생성합니다.
"""

import os
import argparse
import logging
import pandas as pd
from typing import List, Tuple

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS as SparkALS
from utils import setup_logging
from database.recommendation_db import RecommendationDB

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
    PySpark ALS를 사용하여 전체 사용자에 대한 추천을 생성합니다.
    """
    try:
        import findspark
        findspark.init()
        
        # SparkSession 생성
        spark = (SparkSession.builder
            .appName("ALS Recommendation System")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .getOrCreate())
        
        # 로그 레벨 설정
        spark.sparkContext.setLogLevel("ERROR")
        
        # DataFrame 생성 (스키마 없이 직접 생성)
        ratings_df = spark.createDataFrame(
            interactions_df[["member_id", "product_id", "rating"]]
        )
        
        # 학습/테스트 데이터 분할
        train_data, test_data = ratings_df.randomSplit([0.8, 0.2])
        
        # ALS 모델 설정
        als = SparkALS(
            maxIter=15,
            regParam=0.1,
            rank=10,
            userCol="member_id",
            itemCol="product_id",
            ratingCol="rating",
            coldStartStrategy="drop"
        )
        
        # 모델 학습
        model = als.fit(train_data)
        
        # 성능 평가
        predictions = model.transform(test_data)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        print(f"Root-mean-square error (RMSE): {rmse:.2f}")
        
        # 전체 사용자에 대한 추천 생성
        user_recs = model.recommendForAllUsers(top_n)
        
        # recommendations 컬럼을 평탄화
        flattened_recs = user_recs.select(
            "member_id",
            F.explode("recommendations").alias("rec")
        )
        
        # rec 컬럼에서 product_id와 rating 분리
        flattened_recs = flattened_recs.select(
            "member_id",
            flattened_recs["rec.product_id"].alias("product_id"),
            flattened_recs["rec.rating"].alias("predicted_rating")
        )
        
        # Pandas DataFrame으로 변환
        recommendations_df = flattened_recs.toPandas()
        
        # SparkSession 종료
        spark.stop()
        
        return recommendations_df
        
    except Exception as e:
        logging.error(f"PySpark ALS 추천 생성 실패: {str(e)}")
        raise Exception(f"PySpark ALS 오류: {str(e)}")

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
        print("\n=== PySpark ALS 모델로 추천 생성 시작 ===")
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