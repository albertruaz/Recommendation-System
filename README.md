# 🚀 추천 시스템

**모듈화되고 확장 가능한** PySpark ALS 기반 사용자-상품 추천 시스템

## ⚡ 빠른 시작

```bash
# 추천 시스템 실행
python run.py
```

끝! 🎉 모든 결과는 `output/` 폴더에 저장됩니다.

## 📋 주요 기능

- **PySpark ALS 기반 추천**: 대용량 데이터 처리 가능
- **다양한 상호작용 지원**: 조회, 좋아요, 장바구니, 구매, 리뷰 등
- **자동 파일 저장**: CSV 형태로 추천 결과 저장
- **데이터베이스 저장**: PostgreSQL/MySQL 지원 (설정 가능)
- **모델 평가**: MAE, RMSE 자동 계산
- **완전한 로깅**: 실행 과정 상세 기록

## ⚙️ 설정

모든 설정은 `als_config.json` 파일에서 관리됩니다:

```json
{
  "pyspark_als": {
    "max_iter": 10, // 학습 반복 횟수
    "reg_param": 0.1, // 정규화 파라미터
    "rank": 10, // 잠재 요인 차원
    "interaction_weights": {
      "view1": 2.0, // 상호작용별 가중치
      "like": 5.0,
      "cart": 7.0,
      "purchase": 10.0
    }
  },
  "default_params": {
    "days": 30, // 데이터 수집 기간
    "top_n": 10 // 사용자당 추천 개수
  },
  "database": {
    "save_to_db": true, // DB 저장 여부
    "db_type": "postgres" // postgres 또는 mysql
  }
}
```

## 📁 프로젝트 구조

```
recommendation/
├── run.py                 # 🌟 메인 실행 파일
├── als_config.json        # ⚙️ 설정 파일
├── config/               # 설정 관리
├── core/                 # 핵심 로직 (데이터, 모델)
├── services/             # 서비스 레이어 (추천, 저장, DB)
├── database/             # 데이터베이스 연결
├── utils/                # 유틸리티 (로깅, Spark 등)
└── output/               # 📂 결과 저장 폴더
```

## 🔧 환경 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (.env)

```env
# MySQL 연결
DB_HOST=your_db_host
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database

# PostgreSQL 연결
PG_HOST=your_pg_host
PG_USER=your_pg_username
PG_PASSWORD=your_pg_password
PG_DB_NAME=your_pg_database
```

## 💻 사용법

### 기본 실행

```bash
python run.py
```

### 설정 변경 후 실행

1. `als_config.json` 파일 수정
2. `python run.py` 실행

### 프로그래밍 방식

```python
from run import RecommendationRunner

runner = RecommendationRunner()
results = runner.run()

print(f"추천 개수: {results['recommendations_count']}")
print(f"출력 폴더: output/{results['run_id']}")
```

## 📊 출력 결과

실행 후 `output/{run_id}/` 폴더에 다음 파일들이 생성됩니다:

- `recommendations_{run_id}.csv`: 추천 결과
- `user_factors_{run_id}.csv`: 사용자 잠재 요인
- `item_factors_{run_id}.csv`: 상품 잠재 요인
- `evaluation_results.json`: 모델 평가 결과

### 추천 결과 형태

```csv
member_id,product_id,predicted_rating
12345,67890,8.5
12345,54321,7.9
...
```

## 🎯 구조 개선 이점

### Before (기존)

- `pyspark_als.py` (586줄) - 너무 복잡
- `run_als.py` (224줄) - 책임 분산 안됨
- 여러 실행 파일들 - 혼란스러움

### After (개선)

- `run.py` (156줄) - 단일 진입점
- 모듈화된 구조 - 각 클래스가 50-180줄
- 설정 중심 - 코드 수정 없이 동작 변경
- 확장 가능 - 새 기능 추가 용이

## 🆘 문제 해결

### 자주 발생하는 오류

1. **Spark 메모리 부족**

   ```bash
   export SPARK_DRIVER_MEMORY=4g
   export SPARK_EXECUTOR_MEMORY=4g
   ```

2. **DB 연결 오류**

   - `.env` 파일의 DB 설정 확인
   - 네트워크 연결 상태 확인

3. **데이터 없음 오류**
   - `als_config.json`의 `days` 값 증가
   - 데이터베이스 내 상호작용 데이터 확인

## 🔮 확장 계획

- [ ] 실시간 추천 API
- [ ] A/B 테스트 기능
- [ ] 다양한 추천 알고리즘 추가
- [ ] 웹 대시보드
- [ ] 자동 모델 재학습

---

**문의사항이나 버그 리포트는 이슈를 생성해주세요!** 🐛
