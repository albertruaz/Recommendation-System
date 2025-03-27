# Implicit ALS 기반 추천 시스템

이 프로젝트는 Implicit ALS(Alternating Least Squares) 알고리즘을 사용하여 사용자-상품 추천을 생성하는 시스템입니다.

## 주요 기능

- 사용자-상품 상호작용 데이터를 기반으로 한 추천 생성
- 다양한 상호작용 타입(조회, 좋아요, 장바구니 등)에 대한 가중치 적용
- 상호작용 빈도와 타입을 고려한 신뢰도 점수 계산
- 대규모 사용자-상품 데이터 처리 지원

## 시스템 요구사항

- Python 3.8 이상
- MySQL 데이터베이스

## 설치 방법

1. 저장소 클론

```bash
git clone [repository_url]
cd recommendation
```

2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
   `.env` 파일을 생성하고 다음 내용을 설정:

```
# 데이터베이스 설정
DB_HOST=your_host
DB_PORT=your_port
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_database
```

## 사용 방법

기본 실행:

```bash
python main_als.py
```

옵션 지정:

```bash
python main_als.py --days 30 --top_n 300 --output_dir output --verbose
```

### 주요 매개변수

- `--days`: 최근 몇 일간의 데이터를 사용할지 지정 (기본값: 30)
- `--top_n`: 각 사용자에게 추천할 상품 수 (기본값: 300)
- `--output_dir`: 결과를 저장할 디렉토리 (기본값: output)
- `--verbose`: 상세 로깅 활성화

## 프로젝트 구조

```
recommendation/
├── config/
│   └── rating_weights.py     # 상호작용 가중치 설정
├── database/
│   ├── db_connector.py       # 데이터베이스 연결 관리
│   └── recommendation_db.py  # 추천 관련 데이터베이스 쿼리
├── models/
│   └── als.py               # ALS 추천 시스템 구현
├── utils/
│   ├── __init__.py
│   ├── config.py            # 설정 파일 관리
│   └── logger.py            # 로깅 설정
├── output/                   # 추천 결과 저장 디렉토리
├── logs/                     # 로그 파일 디렉토리
├── .env                      # 환경 변수 설정
├── .gitignore
├── main_als.py              # 메인 실행 스크립트
└── requirements.txt         # 필요한 패키지 목록
```

## 상호작용 가중치

상호작용 타입별 가중치는 `config/rating_weights.py`에 정의되어 있습니다:

- 조회 (view)
  - view_type_1: 3.0 (3회 이상 조회)
  - view_type_2: 2.0 (1회 이상 조회)
  - view_type_3: 1.0 (단순 노출)
- 좋아요 (like): 4.0
- 장바구니 (cart): 5.0
- 구매 (purchase): 10.0
- 리뷰 (review): 8.0

## 로깅

로그는 `logs` 디렉토리에 저장되며, 다음 정보를 포함합니다:

- 데이터베이스 연결/쿼리 실행
- 모델 학습 과정
- 추천 생성 과정
- 오류 및 예외 상황

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
