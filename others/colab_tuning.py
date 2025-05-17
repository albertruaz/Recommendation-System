# -*- coding: utf-8 -*-
"""ALS 모델 하이퍼파라미터 튜닝"""

from google.colab import files
uploaded = files.upload()

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# CSV 파일 읽기
csv_file = list(uploaded.keys())[0]
pdf = pd.read_csv(csv_file)

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("ALS Hyperparameter Tuning") \
    .getOrCreate()

# CSV 데이터를 Spark DataFrame으로 변환
columns = ["member_id", "product_id", "rating"]
ratings_df = spark.createDataFrame(pdf, columns)
ratings_df.show()

# 데이터를 train/test로 분리 (튜닝에 사용할 목적)
train_data, test_data = ratings_df.randomSplit([0.8, 0.2])

# 평가 지표 셋업
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

# 하이퍼파라미터 후보 리스트 (필요에 따라 리스트의 원소를 조정할 수 있음)
maxIter_list = [15]               # 예시: 하나의 값만 사용 가능하도록 제한 (원하는 경우 값을 늘릴 수 있음)
regParam_list = [0.1, 0.01, 0.001]
rank_list = [10, 20, 30]

# 튜닝 결과를 저장할 리스트
tuning_results = []

# 3중 for문을 사용해 하이퍼파라미터 조합별 모델 학습 및 평가
for maxIter in maxIter_list:
    for regParam in regParam_list:
        for rank in rank_list:
            # ALS 모델 초기화
            als = ALS(
                maxIter=maxIter,
                regParam=regParam,
                rank=rank,
                userCol="member_id",
                itemCol="product_id",
                ratingCol="rating",
                coldStartStrategy="drop"
            )
            
            # 모델 학습
            model = als.fit(train_data)
            
            # 테스트 데이터에 대한 예측
            predictions = model.transform(test_data)
            
            # RMSE 평가
            rmse = evaluator.evaluate(predictions)
            
            # 결과 출력 및 저장
            print(f"maxIter: {maxIter}, regParam: {regParam}, rank: {rank} => RMSE: {rmse:.4f}")
            tuning_results.append({
                "maxIter": maxIter,
                "regParam": regParam,
                "rank": rank,
                "rmse": rmse
            })

# 튜닝 결과 확인 (옵션)
import pandas as pd
results_df = pd.DataFrame(tuning_results)
print(results_df)

# Spark 세션 종료
spark.stop()
