"""
CatBoost 기반 하이브리드 랭킹 모델

협업 필터링과 콘텐츠 기반 추천의 결과를 결합하고,
사용자/아이템 특징을 활용하여 개인화된 랭킹을 생성합니다.
"""

import os
import logging
from typing import List, Tuple, Dict, Generator
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from models.base.base_recommender import BaseRecommender

class CatBoostHybridRanker(BaseRecommender):
    """CatBoost 기반 하이브리드 랭킹 모델"""
    
    def __init__(self, config_dict=None):
        """
        Args:
            config_dict (dict, optional): 모델 설정
        """
        super().__init__()
        self.config = config_dict or {}
        self.model_config = {
            'iterations': self.config.get('iterations', 500),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'depth': self.config.get('depth', 6),
            'loss_function': self.config.get('loss_function', 'YetiRank'),
            'verbose': self.config.get('verbose', 100),
            'train_batch_size': self.config.get('train_batch_size', 1000),
            'eval_batch_size': self.config.get('eval_batch_size', 100)
        }
        self.model = None
        self.feature_names = None
        
    def _prepare_features(self, user_id: int, item_ids: List[int], 
                         cf_scores: List[float], content_scores: List[float],
                         item_features: pd.DataFrame = None) -> pd.DataFrame:
        """특성 데이터 준비
        
        Args:
            user_id (int): 사용자 ID
            item_ids (List[int]): 아이템 ID 리스트
            cf_scores (List[float]): 협업 필터링 점수 리스트
            content_scores (List[float]): 콘텐츠 기반 점수 리스트
            item_features (pd.DataFrame, optional): 아이템 특성 데이터
            
        Returns:
            pd.DataFrame: 특성 데이터프레임
        """
        # 기본 특성
        features = pd.DataFrame({
            'user_id': [user_id] * len(item_ids),
            'item_id': item_ids,
            'cf_score': cf_scores,
            'content_score': content_scores
        })
        
        # 아이템 특성 추가
        if item_features is not None and not item_features.empty:
            features = features.merge(item_features, on='item_id', how='left')
            
        return features
        
    def _batch_generator(self, features: pd.DataFrame, batch_size: int) -> Generator:
        """배치 데이터 생성기
        
        Args:
            features (pd.DataFrame): 특성 데이터
            batch_size (int): 배치 크기
            
        Yields:
            Generator: 배치 데이터
        """
        num_samples = len(features)
        indices = np.arange(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield features.iloc[batch_indices]
    
    def train(self, train_features: pd.DataFrame, train_labels: np.ndarray,
              group_ids: np.ndarray, item_features: pd.DataFrame = None) -> None:
        """모델 학습
        
        Args:
            train_features (pd.DataFrame): 학습 특성 데이터
            train_labels (np.ndarray): 학습 레이블
            group_ids (np.ndarray): 그룹 ID (사용자별 그룹핑)
            item_features (pd.DataFrame, optional): 아이템 특성 데이터
            
        Raises:
            ValueError: 데이터 형식이 잘못된 경우
            RuntimeError: 학습 중 오류 발생 시
        """
        try:
            logging.info("CatBoost 하이브리드 랭킹 모델 학습 시작")
            
            if train_features.empty or len(train_labels) == 0 or len(group_ids) == 0:
                raise ValueError("빈 데이터셋으로 학습할 수 없습니다.")
                
            # 특성 데이터 준비
            if item_features is not None and not item_features.empty:
                train_features = train_features.merge(item_features, on='item_id', how='left')
            
            # 범주형 특성 식별
            cat_features = ['user_id', 'item_id'] + \
                         [col for col in train_features.select_dtypes(include=['object']).columns
                          if col not in ['user_id', 'item_id']]
            
            # 특성 이름 저장
            self.feature_names = list(train_features.columns)
            
            # 모델 초기화
            self.model = CatBoostRanker(
                iterations=self.model_config['iterations'],
                learning_rate=self.model_config['learning_rate'],
                depth=self.model_config['depth'],
                loss_function=self.model_config['loss_function'],
                verbose=self.model_config['verbose'],
                cat_features=cat_features
            )
            
            # 배치 학습
            for batch_features in self._batch_generator(train_features, self.model_config['train_batch_size']):
                batch_indices = batch_features.index
                batch_labels = train_labels[batch_indices]
                batch_groups = group_ids[batch_indices]
                
                train_pool = Pool(
                    data=batch_features,
                    label=batch_labels,
                    group_id=batch_groups,
                    cat_features=cat_features
                )
                
                self.model.fit(train_pool, verbose=self.model_config['verbose'])
            
            self.is_trained = True
            logging.info("CatBoost 하이브리드 랭킹 모델 학습 완료")
            
        except Exception as e:
            self.is_trained = False
            logging.error(f"CatBoost 하이브리드 랭킹 모델 학습 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 학습 실패: {str(e)}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """예측 점수 생성
        
        Args:
            features (pd.DataFrame): 특성 데이터
            
        Returns:
            np.ndarray: 예측 점수
            
        Raises:
            RuntimeError: 모델이 학습되지 않은 경우
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
            
        try:
            predictions = []
            
            # 배치 예측
            for batch_features in self._batch_generator(features, self.model_config['eval_batch_size']):
                batch_predictions = self.model.predict(batch_features)
                predictions.extend(batch_predictions)
                
            return np.array(predictions)
            
        except Exception as e:
            logging.error(f"예측 생성 중 오류 발생: {str(e)}")
            raise RuntimeError(f"예측 실패: {str(e)}")
    
    def get_recommendations(self, user_id: int, item_ids: List[int],
                          cf_scores: List[float], content_scores: List[float],
                          item_features: pd.DataFrame = None, k: int = 10) -> List[Tuple[int, float]]:
        """특정 사용자에 대한 추천 생성
        
        Args:
            user_id (int): 사용자 ID
            item_ids (List[int]): 후보 아이템 ID 리스트
            cf_scores (List[float]): 협업 필터링 점수 리스트
            content_scores (List[float]): 콘텐츠 기반 점수 리스트
            item_features (pd.DataFrame, optional): 아이템 특성 데이터
            k (int): 추천할 아이템 개수
            
        Returns:
            List[Tuple[int, float]]: (아이템 ID, 점수) 튜플의 리스트
            
        Raises:
            RuntimeError: 모델이 학습되지 않은 경우
            ValueError: 입력 데이터 형식이 잘못된 경우
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
            
        if len(item_ids) != len(cf_scores) or len(item_ids) != len(content_scores):
            raise ValueError("입력 리스트의 길이가 일치하지 않습니다.")
            
        try:
            # 특성 데이터 준비
            features = self._prepare_features(
                user_id, item_ids, cf_scores, content_scores, item_features
            )
            
            # 예측
            scores = self.predict(features)
            
            # Top-K 아이템 선택
            top_indices = np.argsort(scores)[-k:][::-1]
            recommendations = [
                (item_ids[idx], float(scores[idx]))
                for idx in top_indices
            ]
            
            return recommendations
            
        except Exception as e:
            logging.error(f"추천 생성 중 오류 발생: {str(e)}")
            raise RuntimeError(f"추천 생성 실패: {str(e)}")
    
    def save_model(self, path: str) -> None:
        """모델 저장
        
        Args:
            path (str): 저장 경로
            
        Raises:
            RuntimeError: 모델이 학습되지 않은 경우
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # CatBoost 모델 저장
            model_path = f"{path}.cbm"
            self.model.save_model(model_path)
            
            # 설정 저장
            config_path = f"{path}.config"
            pd.to_pickle({
                'config': self.config,
                'model_config': self.model_config,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }, config_path)
            
            logging.info(f"모델 저장 완료: {path}")
            
        except Exception as e:
            logging.error(f"모델 저장 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 저장 실패: {str(e)}")
        
    def load_model(self, path: str) -> None:
        """모델 로드
        
        Args:
            path (str): 모델 파일 경로
            
        Raises:
            FileNotFoundError: 모델 파일이 없는 경우
            RuntimeError: 모델 로드 중 오류 발생 시
        """
        model_path = f"{path}.cbm"
        config_path = f"{path}.config"
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
            
        try:
            # CatBoost 모델 로드
            self.model = CatBoostRanker()
            self.model.load_model(model_path)
            
            # 설정 로드
            checkpoint = pd.read_pickle(config_path)
            self.config = checkpoint['config']
            self.model_config = checkpoint['model_config']
            self.feature_names = checkpoint['feature_names']
            self.is_trained = checkpoint['is_trained']
            
            logging.info(f"모델 로드 완료: {path}")
            
        except Exception as e:
            self.is_trained = False
            logging.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 로드 실패: {str(e)}") 