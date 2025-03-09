"""
하이브리드 추천 시스템 실행 스크립트

이 스크립트는 학습된 하이브리드 추천 모델을 사용하여
특정 사용자에 대한 개인화된 추천을 생성합니다.
"""

import argparse
import logging
from functools import lru_cache
from models.hybrid import create_hybrid_model, DEFAULT_HYBRID_MODEL
from models.collaborative.recbole_bpr import BPRModel
from models.content.tfidf import TFIDFModel
from utils import load_and_preprocess_data, setup_logging, get_model_path, load_config

class RecommendationError(Exception):
    """추천 시스템 관련 예외"""
    pass

@lru_cache(maxsize=128)
def get_recommendations(user_id: int, n_recommendations: int = 10, hybrid_model_type=DEFAULT_HYBRID_MODEL):
    """특정 사용자에 대한 추천을 생성합니다. (캐싱 지원)
    
    Args:
        user_id (int): 추천을 받을 사용자 ID
        n_recommendations (int): 추천할 아이템 개수
        hybrid_model_type (str): 사용할 하이브리드 모델 타입
        
    Returns:
        List[Tuple[int, float]]: (아이템 ID, 점수) 형태의 추천 리스트
        
    Raises:
        RecommendationError: 추천 생성 중 오류 발생 시
    """
    try:
        logging.info(f"사용자 {user_id}에 대한 추천 생성 시작 (모델: {hybrid_model_type})")
        
        # 데이터 로드
        train, test, items = load_and_preprocess_data()
        
        # 사용자 ID 유효성 검사
        if user_id not in train['user_id'].unique():
            raise RecommendationError(f"사용자 ID {user_id}가 학습 데이터에 존재하지 않습니다.")
        
        # 설정 로드
        config = load_config()
        
        # 하이브리드 모델 로드
        hybrid_model = create_hybrid_model(model_type=hybrid_model_type, config_dict=config.get(hybrid_model_type, {}))
        hybrid_model.load_model(get_model_path(f"{hybrid_model_type}_hybrid_model"))
        
        # 모델 타입에 따른 추천 생성
        if hybrid_model_type == 'catboost':
            # 기본 모델 로드 (CatBoost는 BPR과 TF-IDF 모델의 점수를 필요로 함)
            bpr_model = BPRModel()
            bpr_model.load_model(get_model_path("bpr_model"))
            
            tfidf_model = TFIDFModel()
            tfidf_model.load_model(get_model_path("tfidf_model"))
            
            # 모든 아이템에 대한 점수 계산
            all_items = items['item_id'].unique().tolist()
            cf_scores = [bpr_model.get_score(user_id, item_id) for item_id in all_items]
            content_scores = []
            for item_id in all_items:
                content_recs = tfidf_model.get_recommendations(item_id, k=10)
                content_dict = {rec_id: score for rec_id, score in content_recs}
                content_scores.append(content_dict.get(item_id, 0))
                
            recommendations = hybrid_model.get_recommendations(
                user_id, all_items, cf_scores, content_scores, item_features=items, k=n_recommendations
            )
        else:
            recommendations = hybrid_model.get_recommendations(user_id, k=n_recommendations)
        
        logging.info(f"사용자 {user_id}에 대한 추천 생성 완료")
        return recommendations
        
    except FileNotFoundError:
        error_msg = f"필요한 모델 파일을 찾을 수 없습니다. 먼저 train.py를 실행하여 {hybrid_model_type} 모델을 학습해주세요."
        logging.error(error_msg)
        raise RecommendationError(error_msg)
        
    except Exception as e:
        error_msg = f"추천 생성 중 오류 발생: {str(e)}"
        logging.error(error_msg)
        raise RecommendationError(error_msg)

def main():
    parser = argparse.ArgumentParser(
        description='하이브리드 추천 시스템',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--user_id', type=int, required=True,
                       help='추천을 받을 사용자 ID')
    parser.add_argument('--n', type=int, default=10,
                       help='추천할 아이템 개수 (기본값: 10)')
    parser.add_argument('--model', type=str, default=DEFAULT_HYBRID_MODEL,
                       help=f'사용할 하이브리드 모델 타입 (기본값: {DEFAULT_HYBRID_MODEL})')
    parser.add_argument('--verbose', action='store_true',
                       help='상세 로깅 활성화')
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        recommendations = get_recommendations(args.user_id, args.n, args.model)
        
        print(f"\n=== 사용자 {args.user_id}를 위한 추천 결과 ({args.model} 모델) ===")
        print("-" * 50)
        for i, (item_id, score) in enumerate(recommendations, 1):
            print(f"{i}. 아이템 {item_id}: {score:.4f}")
            
    except RecommendationError as e:
        print(f"추천 생성 실패: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        logging.exception("상세 오류 정보:")
        exit(1)

if __name__ == "__main__":
    main() 