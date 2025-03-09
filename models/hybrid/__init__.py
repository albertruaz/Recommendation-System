from .catboost_hybrid_ranker import CatBoostHybridRanker

# 하이브리드 모델 타입을 열거형처럼 정의
HYBRID_MODEL_CATBOOST = 'catboost'

# 기본 모델 타입
DEFAULT_HYBRID_MODEL = HYBRID_MODEL_CATBOOST

def create_hybrid_model(model_type=DEFAULT_HYBRID_MODEL, **kwargs):
    """
    하이브리드 모델 팩토리 함수
    
    Args:
        model_type (str): 모델 타입 ('catboost' 등)
        **kwargs: 모델 초기화 인자
        
    Returns:
        BaseRecommender: 생성된 하이브리드 모델 인스턴스
    """
    if model_type == HYBRID_MODEL_CATBOOST:
        return CatBoostHybridRanker(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 하이브리드 모델 타입: {model_type}")

__all__ = ['CatBoostHybridRanker', 'create_hybrid_model', 
           'HYBRID_MODEL_CATBOOST', 'DEFAULT_HYBRID_MODEL'] 