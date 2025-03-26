# ALS 기반 추천 시스템

Implicit ALS와 PySpark ALS를 사용한 사용자-상품 추천 시스템입니다.

## 설치 방법

1. 가상환경 생성 및 활성화:

```bash
conda create -n recsys python=3.8
conda activate recsys
```

2. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 데이터베이스에서 추천 생성

```bash
python main_als.py --mode implicit --user_id [USER_ID] --data_source db --days [DAYS]
```

예시:

```bash
python main_als.py --mode implicit --user_id 123 --data_source db --days 30
```

### CSV 파일에서 추천 생성

```bash
python main_als.py --mode implicit --user_id [USER_ID] --data_source csv --csv_path [CSV_PATH]
```

예시:

```bash
python main_als.py --mode implicit --user_id 123 --data_source csv --csv_path data/interactions.csv
```

### 입력 데이터 형식

CSV 파일을 사용할 경우, 다음 컬럼이 필요합니다:

- `member_id`: 사용자 ID
- `product_id`: 상품 ID
- `rating`: 평점 (없을 경우 자동으로 1로 설정)

### 주요 파라미터

- `--mode`: 'implicit' 또는 'pyspark' (기본값: implicit)
- `--user_id`: 추천을 받을 사용자 ID
- `--n`: 추천할 상품 개수 (기본값: 10)
- `--days`: 최근 몇 일간의 데이터를 사용할지 (기본값: 90)
- `--data_source`: 'db' 또는 'csv'
- `--csv_path`: CSV 파일 경로
- `--output_csv`: 결과를 저장할 CSV 파일 경로

### 모델 설정

`configs/model_config.json` 파일에서 ALS 모델의 파라미터를 조정할 수 있습니다:

- `factors`: 잠재 요인 수
- `regularization`: 정규화 계수
- `iterations`: 반복 횟수
- `alpha`: 신뢰도 가중치

## License

MIT License
