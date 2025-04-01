# Implicit ALS 기반 추천 시스템

이 프로젝트는 Implicit ALS(Alternating Least Squares) 알고리즘을 사용하여 사용자-상품 추천을 생성하는 시스템입니다.

## 주요 기능

### 1. 상호작용 기반 추천

- 다양한 사용자-상품 상호작용 데이터 활용
  - 조회 (view): 조회 횟수와 타입에 따른 가중치 적용
  - 좋아요 (like): 상품에 대한 관심도 반영
  - 장바구니 (cart): 구매 의도 반영
  - 구매 (purchase): 실제 구매 행동 반영
  - 리뷰 (review): 상품 사용 후기 반영

### 2. 가중치 시스템

- 상호작용 타입별 차등 가중치 적용
  - 조회 타입별 가중치: 1.0 ~ 3.0
  - 좋아요: 5.0
  - 장바구니: 10.0
  - 구매: 13.0
  - 리뷰: 15.0

### 3. 자동 로깅 시스템

- 날짜별 로그 파일 자동 생성
- 모든 실행 정보와 에러를 로그 파일에 기록
- 로그 레벨별 구분된 메시지 관리

## 환경 설정

### 1. Conda 환경 생성 및 활성화

```bash
# 환경 생성
conda create -n recommendation python=3.8
conda activate recommendation

# 필요한 패키지 설치
conda install -c conda-forge pandas numpy scipy
conda install -c conda-forge implicit
conda install -c conda-forge pymysql sqlalchemy python-dotenv
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 설정:

```env
# 데이터베이스 설정
DB_HOST=your_host
DB_PORT=your_port
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_database
```

## 설정 파일 (config/config.json)

### 1. 상호작용 가중치

```json
"interaction_weights": {
  "view_type_1": 3.0,  // 3회 이상 조회
  "view_type_2": 2.0,  // 1회 이상 조회
  "view_type_3": 1.0,  // 단순 노출
  "like": 5.0,         // 좋아요
  "cart": 10.0,        // 장바구니
  "purchase": 13.0,    // 구매
  "review": 15.0       // 리뷰
}
```

### 2. 모델 파라미터

```json
"als_params": {
  "max_iter": 15,      // 최대 반복 횟수
  "reg_param": 0.1,    // 정규화 파라미터
  "rank": 10,          // 잠재 요인 개수
}
```

### 3. 기본 실행 설정

```json
"default_params": {
  "days": 30,          // 최근 데이터 기간
  "top_n": 300,        // 추천 상품 수
  "output_dir": "output"  // 결과 저장 경로
}
```

## 실행 방법

기본 실행:

```bash
python main_als.py
```

## 프로젝트 구조

```
recommendation/
├── config/
│   └── config.json           # 설정 파일
├── database/
│   ├── db_connector.py       # DB 연결 관리
│   └── recommendation_db.py  # 추천 관련 DB 쿼리
├── models/
│   ├── base_als.py          # ALS 기본 클래스
│   └── implicit_als.py      # Implicit ALS 구현
├── utils/
│   └── logger.py            # 로깅 설정
├── logs/                    # 로그 파일 저장
├── output/                  # 추천 결과 저장
├── .env                     # 환경 변수
└── main_als.py             # 메인 실행 스크립트
```

## 출력 결과

추천 결과는 CSV 파일로 저장되며 다음 정보를 포함합니다:

- member_id: 사용자 ID
- product_id: 추천 상품 ID
- predicted_rating: 예측 선호도 점수

## 로깅

모든 실행 로그는 `logs` 디렉토리에 날짜별로 저장됩니다:

- 파일명 형식: YYYYMMDD.log
- 로그 포맷: `시간 - 모듈명 - 로그레벨 - 메시지`
