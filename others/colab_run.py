# -*- coding: utf-8 -*-
"""추천 상품 추출 (ALS 모델, 학습 데이터 제외 추천)"""

from google.colab import files
uploaded = files.upload()

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("ALS Recommendation with Filtered Items") \
    .getOrCreate()

# CSV 파일 읽기
csv_file = list(uploaded.keys())[0]
pdf = pd.read_csv(csv_file)

# CSV 데이터를 Spark DataFrame으로 변환 (컬럼명: member_id, product_id, rating)
columns = ["member_id", "product_id", "rating"]
ratings_df = spark.createDataFrame(pdf, columns)
ratings_df.printSchema()
ratings_df.show()

# 전체 데이터를 학습 데이터로 사용 (train/test 분리 없이 학습)
train_data = ratings_df

# explicit feedback 기반 ALS 모델 초기화 및 학습
als = ALS(
    maxIter=15,
    regParam=0.1,
    rank=10,
    userCol="member_id",
    itemCol="product_id",
    ratingCol="rating",
    coldStartStrategy="drop"  # 학습에 없는 아이템/사용자 예측 시 해당 결과 제외
)

model = als.fit(train_data)

# 사용자별 추천 결과 생성 (각 사용자당 300개 추천)
user_recs = model.recommendForAllUsers(300)

# 추천 결과 평탄화 (flattening): recommendations 배열을 explode를 통해 행 단위로 분리
flattened_recs = user_recs.select(
    "member_id",
    F.explode("recommendations").alias("rec")
).select(
    "member_id",
    F.col("rec.product_id").alias("product_id"),
    F.col("rec.rating").alias("rating")
)

# 학습에 사용된 (평가하거나 상호작용한) (member_id, product_id) 쌍을 추출
train_user_items = train_data.select("member_id", "product_id").dropDuplicates()

# left_anti join을 사용하여 이미 학습에 사용된 아이템은 추천 결과에서 제외
filtered_recs = flattened_recs.join(train_user_items, on=["member_id", "product_id"], how="left_anti")

# 최종 추천 결과 출력 (각 사용자에 대해 학습 데이터에 없는 추천 아이템들)
filtered_recs.show(truncate=False)

# (옵션) 추천 결과를 CSV 파일로 저장
filtered_recs.coalesce(1).write.csv('/content/filtered_user_recs.csv', header=True, mode='overwrite')

# Spark 세션 종료
spark.stop()
