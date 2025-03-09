import json
import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def setup_logging():
    """로깅 설정을 초기화합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('recommendation.log')
        ]
    )

def load_config(config_name: str = "model_config.json") -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    config_path = os.path.join("configs", config_name)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"설정 파일 '{config_name}' 로드 완료")
        return config
    except FileNotFoundError:
        logging.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"설정 파일 형식이 잘못되었습니다: {config_path}")
        raise

def ensure_directory(directory: str):
    """디렉토리가 존재하는지 확인하고 없으면 생성합니다."""
    os.makedirs(directory, exist_ok=True)

def setup_directories():
    """필요한 디렉토리들을 생성합니다."""
    directories = [
        "data/raw",
        "data/processed",
        "models/saved"
    ]
    for directory in directories:
        ensure_directory(directory)

def get_model_path(model_name: str) -> str:
    """모델 저장 경로를 반환합니다."""
    return os.path.join("models", "saved", f"{model_name}")

def get_data_path(filename: str, is_raw: bool = True) -> str:
    """데이터 파일 경로를 반환합니다."""
    subdir = "raw" if is_raw else "processed"
    return os.path.join("data", subdir, filename)

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """데이터를 로드하고 전처리합니다."""
    logging.info("데이터 로드 및 전처리 시작")
    config = load_config()["data_preprocessing"]
    
    try:
        # 데이터 로드
        ratings = pd.read_csv(get_data_path("ratings.csv"))
        items = pd.read_csv(get_data_path("item_features.csv"))
        
        logging.info(f"원본 데이터 크기 - ratings: {len(ratings)}, items: {len(items)}")
        
        # 전처리
        ratings.dropna(subset=['user_id', 'item_id'], inplace=True)
        
        # ID 매핑
        user2idx = {uid: idx for idx, uid in enumerate(ratings['user_id'].unique())}
        item2idx = {iid: idx for idx, iid in enumerate(ratings['item_id'].unique())}
        
        ratings['user_id'] = ratings['user_id'].map(user2idx)
        ratings['item_id'] = ratings['item_id'].map(item2idx)
        items['item_id'] = items['item_id'].map(item2idx)
        
        logging.info(f"전처리 후 데이터 크기 - ratings: {len(ratings)}, items: {len(items)}")
        
        # 학습/테스트 분할
        train, test = train_test_split(
            ratings,
            test_size=config['test_size'],
            random_state=config['random_state']
        )
        
        logging.info(f"데이터 분할 - 학습: {len(train)}, 테스트: {len(test)}")
        
        # 처리된 데이터 저장
        train.to_csv(get_data_path("train.csv", is_raw=False), index=False)
        test.to_csv(get_data_path("test.csv", is_raw=False), index=False)
        
        return train, test, items
        
    except Exception as e:
        logging.error(f"데이터 처리 중 오류 발생: {str(e)}")
        raise

def prepare_catboost_features(user_id, item_ids, cf_model, content_model):
    """CatBoost 모델에 사용할 특성 데이터를 준비합니다."""
    features = []
    
    # CF 점수 계산
    cf_scores = [cf_model.get_score(user_id, item_id) for item_id in item_ids]
    
    # 콘텐츠 점수 계산
    content_scores = []
    for item_id in item_ids:
        recs = content_model.get_recommendations(item_id, k=10)
        rec_dict = {rec_id: score for rec_id, score in recs}
        content_scores.append(rec_dict.get(item_id, 0))
    
    # 특성 데이터 생성
    for i, item_id in enumerate(item_ids):
        features.append({
            'user_id': user_id,
            'item_id': item_id,
            'cf_score': cf_scores[i],
            'content_score': content_scores[i]
        })
    
    return pd.DataFrame(features) 