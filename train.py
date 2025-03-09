"""
하이브리드 추천 시스템 학습 스크립트

이 스크립트는 협업 필터링(BPR)과 콘텐츠 기반(TF-IDF) 모델을 결합한
하이브리드 추천 시스템을 학습합니다.
"""

import numpy as np
import time
import logging
import pandas as pd
import argparse
from utils import setup_directories, load_and_preprocess_data, setup_logging, prepare_catboost_features, load_config
from models.collaborative.recbole_bpr import BPRModel
from models.content.tfidf import TFIDFModel
from models.hybrid import create_hybrid_model, HYBRID_MODEL_CATBOOST, DEFAULT_HYBRID_MODEL
from sklearn.metrics import ndcg_score

def evaluate_model(model, test_data, k=10):
    """모델 성능을 평가합니다.
    
    Args:
        model: 평가할 추천 모델
        test_data: 테스트 데이터셋
        k: 추천 아이템 개수
        
    Returns:
        dict: 평가 지표 (NDCG@K, Precision@K, Coverage)
    """
    metrics = {
        'ndcg@k': [],
        'precision@k': [],
        'coverage': set()
    }
    
    test_users = test_data['user_id'].unique()
    for user_id in test_users:
        true_items = test_data[test_data['user_id'] == user_id]['item_id'].tolist()
        if true_items:
            recommendations = model.get_recommendations(user_id, k=k)
            rec_items = [item_id for item_id, _ in recommendations]
            
            # NDCG@K
            relevance = [1 if item in true_items else 0 for item in rec_items]
            metrics['ndcg@k'].append(ndcg_score([relevance], [np.ones_like(relevance)]))
            
            # Precision@K
            metrics['precision@k'].append(len(set(rec_items) & set(true_items)) / k)
            
            # Coverage
            metrics['coverage'].update(rec_items)
    
    return {
        'ndcg@k': np.mean(metrics['ndcg@k']),
        'precision@k': np.mean(metrics['precision@k']),
        'coverage': len(metrics['coverage'])
    }

def prepare_ranker_data(ratings, cf_model, content_model, items):
    """CatBoost 랭킹 모델을 위한 학습 데이터를 준비합니다."""
    features = []
    labels = []
    groups = []
    
    for user_id in ratings['user_id'].unique():
        user_items = ratings[ratings['user_id'] == user_id]['item_id'].tolist()
        if not user_items:
            continue
            
        # 각 아이템에 대한 CF, 콘텐츠 점수 계산
        candidate_items = set(ratings['item_id'].unique())
        user_features = []
        user_labels = []
        
        for item_id in user_items:
            if item_id in candidate_items:
                # CF 점수
                cf_score = cf_model.get_score(user_id, item_id)
                
                # 콘텐츠 점수
                content_recs = content_model.get_recommendations(item_id, k=10)
                content_scores = {rec_id: score for rec_id, score in content_recs}
                
                # 사용자가 상호작용한 아이템 = 양성 샘플
                item_features = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'cf_score': cf_score,
                    'content_score': content_scores.get(item_id, 0),
                    'label': 1.0  # 양성 샘플
                }
                user_features.append(item_features)
                user_labels.append(1.0)
                
                # 사용자가 상호작용하지 않은 아이템 = 음성 샘플
                neg_items = list(candidate_items - set(user_items))
                np.random.shuffle(neg_items)
                for neg_item in neg_items[:5]:  # 각 사용자당 5개 음성 샘플
                    neg_cf_score = cf_model.get_score(user_id, neg_item)
                    neg_content_score = content_scores.get(neg_item, 0)
                    
                    neg_item_features = {
                        'user_id': user_id,
                        'item_id': neg_item,
                        'cf_score': neg_cf_score,
                        'content_score': neg_content_score,
                        'label': 0.0  # 음성 샘플
                    }
                    user_features.append(neg_item_features)
                    user_labels.append(0.0)
        
        if user_features:
            features.extend(user_features)
            labels.extend(user_labels)
            groups.append(len(user_features))
    
    return {
        'features': pd.DataFrame(features),
        'labels': np.array(labels),
        'groups': np.array(groups)
    }

def train_models(hybrid_model_type=DEFAULT_HYBRID_MODEL):
    """모든 모델을 순차적으로 학습합니다.
    
    Args:
        hybrid_model_type (str): 사용할 하이브리드 모델 타입
    """
    start_time = time.time()
    logging.info("모델 학습 시작")
    
    # 1. 데이터 준비
    logging.info("데이터 로드 및 전처리 중...")
    train, test, items = load_and_preprocess_data()
    
    # 2. 협업 필터링 모델 (BPR) 학습
    logging.info("BPR 모델 학습 중...")
    bpr_model = BPRModel()
    bpr_model.train(train)
    bpr_model.save_model("models/saved/bpr_model")
    
    # 3. 콘텐츠 기반 모델 (TF-IDF) 학습
    logging.info("TF-IDF 모델 학습 중...")
    tfidf_model = TFIDFModel()
    tfidf_model.train(items)
    tfidf_model.save_model("models/saved/tfidf_model")
    
    # 4. 하이브리드 모델 학습
    logging.info(f"{hybrid_model_type} 하이브리드 모델 학습 중...")
    
    # 팩토리 함수로 하이브리드 모델 생성
    config = load_config()
    hybrid_model = create_hybrid_model(model_type=hybrid_model_type, config_dict=config.get('catboost', {}))
    
    # 하이브리드 모델 타입에 따른 학습 처리
    if hybrid_model_type == HYBRID_MODEL_CATBOOST:
        # 학습 데이터 준비
        ranker_train_data = prepare_ranker_data(train, bpr_model, tfidf_model, items)
        
        hybrid_model.train(
            train_features=ranker_train_data['features'],
            train_labels=ranker_train_data['labels'],
            group_ids=ranker_train_data['groups'],
            item_features=items
        )
    
    hybrid_model.save_model(f"models/saved/{hybrid_model_type}_hybrid_model")
    
    # 5. 성능 평가
    logging.info("모델 성능 평가 중...")
    hybrid_metrics = evaluate_model(hybrid_model, test)
    
    print("\n=== 모델 성능 평가 결과 ===")
    print(f"{hybrid_model_type.capitalize()} 하이브리드 모델:")
    print(f"NDCG@10: {hybrid_metrics['ndcg@k']:.4f}")
    print(f"Precision@10: {hybrid_metrics['precision@k']:.4f}")
    print(f"아이템 커버리지: {hybrid_metrics['coverage']}")
    
    # 6. 샘플 추천 결과
    print("\n=== 샘플 추천 결과 ===")
    test_users = test['user_id'].unique()[:5]  # 5명만 표시
    
    for user_id in test_users:
        print(f"\n사용자 {user_id}의 추천 결과:")
        
        # 하이브리드 모델 타입에 따른 추천 생성
        if hybrid_model_type == HYBRID_MODEL_CATBOOST:
            # CatBoost 모델을 위한 특성 데이터 준비
            all_items = items['item_id'].unique().tolist()
            cf_scores = [bpr_model.get_score(user_id, item_id) for item_id in all_items]
            content_scores = []
            for item_id in all_items:
                content_recs = tfidf_model.get_recommendations(item_id, k=10)
                content_dict = {rec_id: score for rec_id, score in content_recs}
                content_scores.append(content_dict.get(item_id, 0))
                
            hybrid_recommendations = hybrid_model.get_recommendations(
                user_id, all_items, cf_scores, content_scores, item_features=items, k=5
            )
        else:
            hybrid_recommendations = hybrid_model.get_recommendations(user_id, k=5)
            
        for i, (item_id, score) in enumerate(hybrid_recommendations, 1):
            print(f"  {i}. 아이템 {item_id}: {score:.4f}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"전체 학습 완료 (소요 시간: {elapsed_time:.2f}초)")

if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(
        description='하이브리드 추천 시스템 학습',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, default=DEFAULT_HYBRID_MODEL,
                      help=f'사용할 하이브리드 모델 타입 (기본값: {DEFAULT_HYBRID_MODEL})')
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    
    # 필요한 디렉토리 생성
    setup_directories()
    
    # 모델 학습
    train_models(hybrid_model_type=args.model) 