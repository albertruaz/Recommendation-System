# ALS 기반 추천 시스템

PySpark의 ALS(Alternating Least Squares) 알고리즘을 사용한 사용자-상품 추천 시스템입니다.

## 설치 방법

```bash
# 가상환경 생성 및 패키지 설치
python -m venv recsys && \
source recsys/bin/activate && \
pip install -r requirements.txt && \
pip install findspark pyarrow==14.0.1
```

## 사용 방법

```bash
python main_als.py --days 30 --top_n 300 --output_dir output --verbose
```

### 매개변수 설명

- `--days`: 최근 몇 일간의 데이터를 사용할지 지정 (기본값: 30)
- `--top_n`: 각 사용자에게 추천할 상품 수 (기본값: 300)
- `--output_dir`: 결과를 저장할 디렉토리 (기본값: output)
- `--verbose`: 상세 로깅 활성화

## 모델 설정

### 데이터베이스 설정 (configs/model_config.json)

```json
{
  "data_preprocessing": {
    "test_size": 0.2,
    "random_state": 42,
    "default_days": 30
  },
  "rating_rules": {
    "purchase": {
      "value": 5.0,
      "threshold": 1
    },
    "cart": {
      "value": 4.0,
      "threshold": 1
    },
    "like": {
      "value": 3.0,
      "threshold": 1
    },
    "view": {
      "value": 2.0,
      "threshold": 3
    },
    "impression": {
      "value": 1.0,
      "threshold": 5
    }
  },
  "filter_conditions": {
    "min_interactions": 10,
    "required_actions": ["view", "impression"]
  }
}
```

### ALS 모델 설정

- `max_iter (int)`: 최대 반복 횟수 (기본값: 15)

  - ALS 알고리즘의 반복 횟수를 지정합니다.
  - 값이 클수록 더 정확한 결과를 얻을 수 있지만, 학습 시간이 증가합니다.

- `reg_param (float)`: 정규화 파라미터 (기본값: 0.1)

  - 과적합을 방지하기 위한 정규화 강도를 조절합니다.
  - 값이 클수록 모델이 더 단순해지고, 작을수록 더 복잡한 패턴을 학습합니다.

- `rank (int)`: 잠재 요인 개수 (기본값: 10)

  - 사용자와 상품을 표현하는 잠재 벡터의 차원을 지정합니다.
  - 값이 클수록 더 복잡한 관계를 표현할 수 있지만, 과적합의 위험이 증가합니다.

- `cold_start_strategy (str)`: 콜드 스타트 처리 전략 (기본값: "drop")
  - 새로운 사용자나 상품에 대한 처리 방법을 지정합니다.
  - "drop": 학습 데이터에 없는 사용자/상품은 무시합니다.
  - "nan": 예측값을 NaN으로 설정합니다.

## 코드 설명

### main_als.py

- `get_data_from_db(days: int)`:

  - 데이터베이스에서 최근 N일간의 사용자-상품 상호작용 데이터를 가져옵니다.
  - 상호작용이 없는 경우 예외를 발생시킵니다.

- `generate_recommendations(interactions_df: pd.DataFrame, top_n: int)`:

  - ALS 모델을 초기화하고 학습시킨 후 추천을 생성합니다.
  - RMSE 점수를 출력하고 추천 결과를 반환합니다.
  - 리소스 정리를 보장하기 위해 finally 블록에서 cleanup을 수행합니다.

- `main()`:
  - 커맨드 라인 인자를 파싱하고 로깅을 설정합니다.
  - 데이터를 가져오고 추천을 생성한 후 결과를 저장합니다.
  - 오류 처리와 상세한 로깅을 제공합니다.

### models/als.py (ALSRecommender 클래스)

- `__init__()`:

  - ALS 모델의 하이퍼파라미터를 초기화합니다.
  - SparkSession과 모델 인스턴스를 None으로 초기화합니다.

- `_initialize_spark()`:

  - SparkSession을 생성하고 메모리 설정을 구성합니다.
  - 로그 레벨을 ERROR로 설정하여 불필요한 출력을 제한합니다.

- `_prepare_data(interactions_df: pd.DataFrame)`:

  - 입력 데이터를 SparkDataFrame으로 변환합니다.
  - 학습/테스트 세트로 분할합니다. (8:2 비율)

- `train(interactions_df: pd.DataFrame)`:

  - ALS 모델을 학습시키고 RMSE 점수를 계산합니다.
  - 학습된 모델을 인스턴스 변수로 저장합니다.

- `generate_recommendations(top_n: int)`:

  - 학습된 모델을 사용하여 모든 사용자에 대한 추천을 생성합니다.
  - 결과를 평탄화하고 pandas DataFrame으로 변환합니다.

- `cleanup()`:
  - SparkSession을 종료하고 모델 인스턴스를 정리합니다.
  - 메모리 누수를 방지합니다.

## License

MIT License
