# 추천 시스템

다양한 추천 알고리즘을 활용한 사용자-상품 추천 시스템입니다.

## 주요 기능

### 1. ALS 기반 상호작용 추천

- **사용자-상품 상호작용 기반 추천**
  - 조회 (view): 조회 횟수와 타입에 따른 가중치 적용
  - 좋아요 (like): 상품에 대한 관심도 반영
  - 장바구니 (cart): 구매 의도 반영
  - 구매 (purchase): 실제 구매 행동 반영
  - 리뷰 (review): 상품 사용 후기 반영

### 2. 최근 장바구니 상품 기반 추천

- **최근 장바구니 담은 상품 벡터 기반 추천**
  - 사용자가 최근 장바구니에 담은 상품들의 임베딩 벡터 평균으로 유사 상품 추천
  - PostgreSQL의 PGVector 확장을 활용한 벡터 검색
  - 카테고리 필터링을 통한 관련 상품 추천
  - **[업데이트]** 모듈 구조 개선: 별도의 모델 클래스 없이 런 모듈에서 직접 데이터베이스 연결 및 추천 생성 기능 구현

## 환경 설정

### 1. Conda 환경 설정

```bash
# Conda 환경 생성
conda create -n recommendation python=3.9
conda activate recommendation

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (.env)

```
# MySQL 연결 설정
DB_HOST=your_db_host
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database

# PostgreSQL + PGVector 연결 설정
PG_HOST=your_pg_host
PG_PORT=5432
PG_USER=your_pg_username
PG_PASSWORD=your_pg_password
PG_DB_NAME=your_pg_database

# SSH 터널링 (필요한 경우)
PG_SSH_HOST=your_ssh_host
PG_SSH_USERNAME=your_ssh_username
PG_SSH_PKEY_PATH=/path/to/ssh/key
```

## 설정 파일

### 1. ALS 설정 (config/als_config.json)

```json
{
  "model_type": "pyspark_als",
  "pyspark_als": {
    "max_iter": 15,
    "reg_param": 0.001,
    "rank": 40,
    "interaction_weights": {
      "impression1": 1.0,
      "impression2": 0.5,
      "view1": 5.0,
      "view2": 7.0,
      "like": 10.0,
      "cart": 15.0,
      "purchase": 20.0,
      "review": 20.0
    }
  },
  "default_params": {
    "days": 30,
    "top_n": 100,
    "output_dir": "output"
  }
}
```

### 2. 최근 장바구니 상품 추천 설정 (config/recent_product_config.json)

```json
{
  "model_type": "recent_product",
  "default_params": {
    "days": 30,
    "top_n": 100,
    "output_dir": "output",
    "verbose": true
  },
  "recent_product_model": {
    "similarity_threshold": 0.3,
    "min_interactions": 2,
    "max_cart_items": 20,
    "use_category_filter": true,
    "include_similar_categories": false
  }
}
```

## 실행 방법

### ALS 기반 추천 실행:

```bash
python main.py --model=als
```

### 최근 장바구니 상품 기반 추천 실행:

```bash
python main.py --model=recent_product
```

## 프로젝트 구조

```
recommendation/
├── config/
│   ├── als_config.json           # ALS 모델 설정
│   └── recent_product_config.json # 최근 장바구니 추천 설정
├── database/
│   ├── base_connector.py         # 데이터베이스 연결 기본 클래스
│   ├── db_connector.py           # 레거시 DB 연결 관리
│   ├── db_manager.py             # 중앙화된 데이터베이스 관리자
│   ├── excel_db.py               # 엑셀 데이터 로드 유틸리티
│   ├── mysql_connector.py        # MySQL 연결 구현
│   ├── postgres_connector.py     # PostgreSQL 연결 구현
│   ├── db.py      # 추천 관련 DB 쿼리
│   └── vector_db_connector.py    # 벡터 데이터베이스 연결 및 쿼리
├── model_als/
│   ├── base_als.py               # ALS 기본 클래스
│   ├── pyspark_als.py            # PySpark ALS 구현
│   └── implicit_als.py           # Implicit ALS 구현
├── model_recent_product/
│   └── recent_product_model.py   # [레거시] 최근 장바구니 상품 기반 추천 모델
├── run/
│   ├── run_als.py                # ALS 추천 실행 클래스
│   └── run_recent_product.py     # 최근 장바구니 추천 실행 클래스 (데이터 가져오기 및 추천 생성 기능 포함)
├── utils/
│   ├── logger.py                 # 로깅 설정
│   └── recommendation_utils.py   # 추천 유틸리티
├── cache/                        # 데이터 캐싱 디렉토리
├── logs/                         # 로그 파일 저장
├── output/                       # 추천 결과 저장
├── .env                          # 환경 변수
└── main.py                       # 메인 실행 스크립트
```

## 코드 구조 변경사항

### 최근 장바구니 추천 시스템 리팩토링

- 기존: model_recent_product/recent_product_model.py에 있던 벡터 DB 연결 및 추천 로직을 run/run_recent_product.py로 이동
- 변경:
  - 별도의 모델 클래스 없이 RunRecentProduct 클래스에서 직접 데이터베이스 연결 및 추천 생성 기능 구현
  - VectorDBConnector 클래스를 직접 활용하여 벡터 연산 및 유사도 검색 수행
  - model_recent_product/recent_product_model.py는 레거시 코드로 유지하고 경고 메시지 추가
  - database/vector_db.py 삭제 및 vector_db_connector.py로 기능 통합

### 데이터베이스 구조 개선

- 기존: 개별 연결 클래스들이 독립적으로 존재
- 변경:
  - BaseConnector 추상 클래스 도입으로 공통 인터페이스 제공
  - MySQLConnector, PostgresConnector로 구체적인 구현 분리
  - DatabaseManager를 통한 중앙 집중식 DB 연결 관리
  - 벡터 검색 기능을 VectorDBConnector로 통합

## 출력 결과

추천 결과는 CSV 파일로 저장되며 다음 정보를 포함합니다:

- member_id: 사용자 ID
- product_id: 추천 상품 ID
- score: 추천 점수 (ALS의 경우 predicted_rating, 최근 장바구니 추천의 경우 유사도 점수)

## 로깅

모든 실행 로그는 `logs` 디렉토리에 날짜별로 저장됩니다:

- 파일명 형식: YYYYMMDD.log
- 로그 포맷: `시간 - 모듈명 - 로그레벨 - 메시지`
