# ALS 기반 추천 시스템

ALS(Alternating Least Squares) 알고리즘을 사용하여 사용자-상품 추천을 생성하는 시스템입니다.

## 주요 기능

### 상호작용 기반 추천

- 조회 (view): 조회 횟수와 타입에 따른 가중치 적용
- 좋아요 (like): 상품에 대한 관심도 반영
- 장바구니 (cart): 구매 의도 반영
- 구매 (purchase): 실제 구매 행동 반영
- 리뷰 (review): 상품 사용 후기 반영

## 환경 설정

### 1. Conda 환경 설정

```bash
# Conda 환경 생성
conda create -n als python=3.9.21
conda activate als

# Buffalo 수동 설치 (소스코드에서 직접 설치)
git clone https://github.com/kakao/buffalo.git
cd buffalo
pip install .
cd ..

# 필요한 패키지 설치
pip install -r requirements.txt
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
