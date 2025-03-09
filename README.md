# RecBole 기반 하이브리드 추천 시스템

## 개요

이 프로젝트는 RecBole을 기반으로 하이브리드 추천 시스템을 구현하고, CatBoost를 활용하여 사용자 맞춤형 추천을 제공합니다. 콘텐츠 기반 추천과 협업 필터링(BPR)을 결합하여 더 정확하고 다양한 추천 결과를 생성하며, CatBoost 랭킹 모델로 최종 추천 순위를 개인화합니다.

## 주요 기능

1. **협업 필터링 (BPR)**: RecBole 기반 BPR 모델로 사용자-아이템 상호작용 패턴 학습
2. **콘텐츠 기반 추천**: TF-IDF 모델로 아이템 특성 기반 유사도 계산
3. **CatBoost 하이브리드 랭킹**: 사용자와 아이템 특성, 모델 점수를 활용한 맞춤형 순위 생성

## 모듈 구조 및 확장성

이 시스템은 팩토리 패턴을 사용하여 다양한 하이브리드 추천 모델을 쉽게 통합할 수 있도록 설계되었습니다. 새로운 하이브리드 모델을 추가하려면 다음 단계를 따르세요:

1. `models/hybrid/` 디렉토리에 새로운 모델 클래스 구현
2. `models/hybrid/__init__.py`에 새 모델 타입 상수 추가 및 `create_hybrid_model()` 함수 업데이트
3. 새 모델이 `BaseRecommender` 인터페이스를 구현하는지 확인

## 시스템 구조

```
recommendation/
├── configs/              # 설정 파일
│   └── model_config.json # 모델 하이퍼파라미터 설정
├── models/               # 모델 정의
│   ├── base/             # 기본 모델 인터페이스
│   ├── collaborative/    # 협업 필터링 모델 (BPR)
│   ├── content/          # 콘텐츠 기반 모델 (TF-IDF)
│   └── hybrid/           # 하이브리드 모델 (CatBoost 등)
├── data/                 # 데이터 저장소
│   ├── raw/              # 원본 데이터
│   └── processed/        # 전처리된 데이터
├── train.py              # 모델 학습 스크립트
├── main.py               # 추천 생성 스크립트
└── utils.py              # 유틸리티 함수
```

## 설치 방법

### 요구 사항

- Python 3.8+
- PyTorch 1.10+
- RecBole 1.1.1+
- CatBoost 1.0.0+
- pandas, numpy, scikit-learn

### 설치

```bash
# 가상 환경 생성 (선택 사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install torch>=1.10.0
pip install recbole>=1.1.1
pip install catboost>=1.0.0
pip install pandas numpy scikit-learn
```

## 사용 방법

### 데이터 준비

시스템은 다음과 같은 형식의 데이터를 필요로 합니다:

1. **평점 데이터** (`data/raw/ratings.csv`):

   - user_id: 사용자 식별자
   - item_id: 아이템 식별자
   - rating: 평점 (선택 사항)
   - timestamp: 타임스탬프 (선택 사항)

2. **아이템 특성 데이터** (`data/raw/item_features.csv`):
   - item_id: 아이템 식별자
   - description: 아이템 설명 텍스트
   - category, style 등: 범주형 특성 (선택 사항)
   - embedding_vector: 사전 학습된 임베딩 (선택 사항)

### 모델 학습

```bash
# 기본 하이브리드 모델(CatBoost) 학습
python train.py

# 특정 모델 지정하여 학습
python train.py --model catboost

# 새로운 하이브리드 모델 추가 시 해당 모델 타입 지정
python train.py --model custom_model_type
```

### 추천 생성

```bash
# 명령행에서 추천 생성
python main.py --user_id 123 --n 10 --model catboost

# 추가 옵션
python main.py --user_id 123 --n 10 --model catboost --verbose
```

### 코드에서 추천 생성

```python
from models.hybrid import create_hybrid_model
from models.collaborative.recbole_bpr import BPRModel
from models.content.tfidf import TFIDFModel
from utils import load_and_preprocess_data, get_model_path

# 데이터 로드
train, test, items = load_and_preprocess_data()

# 기본 모델 로드
bpr_model = BPRModel()
bpr_model.load_model(get_model_path("bpr_model"))

tfidf_model = TFIDFModel()
tfidf_model.load_model(get_model_path("tfidf_model"))

# 하이브리드 모델 생성 및 로드
hybrid_model = create_hybrid_model(model_type='catboost')
hybrid_model.load_model(get_model_path("catboost_hybrid_model"))

# 추천을 위한 특성 데이터 준비
user_id = 123
all_items = items['item_id'].unique().tolist()
cf_scores = [bpr_model.get_score(user_id, item_id) for item_id in all_items]
content_scores = []
for item_id in all_items:
    content_recs = tfidf_model.get_recommendations(item_id, k=10)
    content_dict = {rec_id: score for rec_id, score in content_recs}
    content_scores.append(content_dict.get(item_id, 0))

# 추천 생성
recommendations = hybrid_model.get_recommendations(
    user_id, all_items, cf_scores, content_scores, item_features=items, k=5
)

# 결과 출력
for rank, (item_id, score) in enumerate(recommendations, 1):
    print(f"{rank}. 아이템 {item_id} (점수: {score:.4f})")
```

## 하이브리드 모델 확장 방법

새로운 하이브리드 추천 알고리즘을 추가하려면:

1. `models/hybrid/` 디렉토리에 새 모델 클래스 파일 생성 (예: `models/hybrid/my_new_model.py`)
2. `BaseRecommender` 상속 및 필수 메서드 구현:

   ```python
   from models.base.base_recommender import BaseRecommender

   class MyNewHybridModel(BaseRecommender):
       def __init__(self, config_dict=None):
           super().__init__()
           # 초기화 코드

       def train(self, *args, **kwargs):
           # 학습 코드 구현

       def get_recommendations(self, user_id, k=10):
           # 추천 생성 구현

       def save_model(self, path):
           # 모델 저장 구현

       def load_model(self, path):
           # 모델 로드 구현
   ```

3. `models/hybrid/__init__.py` 파일에 새 모델 추가:

   ```python
   from .my_new_model import MyNewHybridModel

   # 상수 추가
   HYBRID_MODEL_NEW = 'my_new_model'

   # 팩토리 함수 업데이트
   def create_hybrid_model(model_type=DEFAULT_HYBRID_MODEL, **kwargs):
       if model_type == HYBRID_MODEL_CATBOOST:
           return CatBoostHybridRanker(**kwargs)
       elif model_type == HYBRID_MODEL_NEW:
           return MyNewHybridModel(**kwargs)
       else:
           raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

   # __all__ 업데이트
   __all__ = [..., 'HYBRID_MODEL_NEW', 'MyNewHybridModel']
   ```

## 성능 평가

시스템은 다음과 같은 지표로 평가됩니다:

- **NDCG@K**: 추천 순위의 품질
- **Precision@K**: 추천 정확도
- **Coverage**: 추천 아이템의 다양성

## 확장 계획

1. 하이퍼파라미터 최적화 자동화
2. 실시간 추천 API 구현
3. 다양한 하이브리드 모델 추가 (LightGBM, Neural Network 등)
4. 멀티모달 특성 지원 (이미지, 텍스트)
5. 설명 가능한 추천 기능 추가

## 저자

본 프로젝트는 하이브리드 추천 시스템과 개인화 랭킹에 관심 있는 개발자들을 위해 개발되었습니다.

## 라이센스

MIT 라이센스
