from .catboost_hybrid_ranker import CatBoostHybridRanker
from .als_hybrid import ALSHybridModel

# 하이브리드 모델 타입을 열거형처럼 정의
HYBRID_MODEL_CATBOOST = 'catboost'
HYBRID_MODEL_ALS = 'als'

# 기본 모델 타입
DEFAULT_HYBRID_MODEL = HYBRID_MODEL_ALS

def create_hybrid_model(model_type=DEFAULT_HYBRID_MODEL, config_dict=None, **kwargs):
    """
    하이브리드 모델 팩토리 함수
    
    Args:
        model_type (str): 모델 타입 ('catboost', 'als' 등)
        config_dict (dict, optional): 모델 설정
        **kwargs: 모델 초기화 인자
        
    Returns:
        BaseRecommender: 생성된 하이브리드 모델 인스턴스
    """
    if model_type == HYBRID_MODEL_CATBOOST:
        return CatBoostHybridRanker(config_dict=config_dict, **kwargs)
    elif model_type == HYBRID_MODEL_ALS:
        return ALSHybridModel(config_dict=config_dict, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 하이브리드 모델 타입: {model_type}")

__all__ = ['CatBoostHybridRanker', 'ALSHybridModel', 'create_hybrid_model', 
           'HYBRID_MODEL_CATBOOST', 'HYBRID_MODEL_ALS', 'DEFAULT_HYBRID_MODEL'] 