"""
RecBole 기반 DSSM(Deep Structured Semantic Model) 콘텐츠 기반 추천 모델

카테고리, 스타일, 임베딩 벡터 등의 아이템 특징을 활용하여
콘텐츠 기반 추천을 수행합니다.
"""

import os
import logging
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from recbole.model.general_recommender import DSSM as RecBoleDSSM
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from models.base.base_recommender import BaseRecommender

class ContentBasedModel(BaseRecommender):
    """RecBole DSSM 모델 래퍼 클래스"""
    
    def __init__(self, config_dict=None):
        """
        Args:
            config_dict (dict, optional): 모델 설정
        """
        super().__init__()
        self.config = config_dict or {}
        self.model_config = {
            'mlp_hidden_size': self.config.get('mlp_hidden_size', [128, 64, 32]),
            'dropout_prob': self.config.get('dropout_prob', 0.1),
            'learning_rate': self.config.get('learning_rate', 0.001),
            'n_epochs': self.config.get('n_epochs', 100),
            'batch_size': self.config.get('batch_size', 256)
        }
        
    def _prepare_data(self, ratings: pd.DataFrame, item_features: pd.DataFrame = None):
        """RecBole 데이터셋 준비 (임시 디렉토리 사용)
        
        Args:
            ratings (pd.DataFrame): 평점 데이터
            item_features (pd.DataFrame, optional): 아이템 특성 데이터
            
        Returns:
            tuple: (config, train_data, valid_data, test_data)
            
        Raises:
            ValueError: 데이터 형식이 잘못된 경우
        """
        if ratings.empty:
            raise ValueError("빈 데이터셋으로 학습할 수 없습니다.")
            
        required_columns = {'user_id', 'item_id'}
        if not required_columns.issubset(ratings.columns):
            raise ValueError(f"필수 컬럼이 없습니다: {required_columns}")
            
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        try:
            # 임시 파일에 데이터 저장
            ratings_file = os.path.join(temp_dir, 'ratings.csv')
            ratings.to_csv(ratings_file, index=False)
            
            if item_features is not None:
                features_file = os.path.join(temp_dir, 'item_features.csv')
                item_features.to_csv(features_file, index=False)
            
            # RecBole 설정
            config = {
                'MODEL_TYPE': 'DSSM',
                'data_path': temp_dir,
                'dataset': 'temp_dataset',
                'load_col': {
                    'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                    'item': ['item_id'] + (list(item_features.columns) if item_features is not None else [])
                },
                'mlp_hidden_size': self.model_config['mlp_hidden_size'],
                'dropout_prob': self.model_config['dropout_prob'],
                'learning_rate': self.model_config['learning_rate'],
                'epochs': self.model_config['n_epochs'],
                'train_batch_size': self.model_config['batch_size'],
                'eval_step': 1
            }
            
            # 데이터셋 생성
            config = Config(model='DSSM', dataset='temp_dataset', config_dict=config)
            dataset = create_dataset(config)
            
            # 데이터 분할 및 처리
            train_data, valid_data, test_data = data_preparation(config, dataset)
            
            return config, train_data, valid_data, test_data
            
        finally:
            # 임시 디렉토리 삭제
            shutil.rmtree(temp_dir)
        
    def train(self, ratings: pd.DataFrame, item_features: pd.DataFrame = None) -> None:
        """모델 학습
        
        Args:
            ratings (pd.DataFrame): 학습 데이터
            item_features (pd.DataFrame, optional): 아이템 특성 데이터
            
        Raises:
            ValueError: 데이터 형식이 잘못된 경우
            RuntimeError: 학습 중 오류 발생 시
        """
        try:
            logging.info("DSSM 모델 학습 시작")
            
            # 데이터 준비
            config, train_data, valid_data, test_data = self._prepare_data(ratings, item_features)
            
            # 모델 초기화
            self.model = RecBoleDSSM(config, train_data.dataset)
            
            # 학습
            trainer = Trainer(config, self.model)
            trainer.fit(train_data, valid_data, saved=True, show_progress=True)
            
            self.is_trained = True
            logging.info("DSSM 모델 학습 완료")
            
        except Exception as e:
            self.is_trained = False
            logging.error(f"DSSM 모델 학습 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 학습 실패: {str(e)}")
    
    def get_recommendations(self, user_id: int, k: int = 10) -> list:
        """특정 사용자에 대한 추천 생성
        
        Args:
            user_id (int): 사용자 ID
            k (int): 추천할 아이템 개수
            
        Returns:
            list: (아이템 ID, 점수) 튜플의 리스트
            
        Raises:
            RuntimeError: 모델이 학습되지 않은 경우
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
            
        try:
            # 추론 모드
            self.model.eval()
            with torch.no_grad():
                # 사용자 임베딩 추출
                user_tensor = torch.tensor([user_id])
                scores = self.model.full_sort_predict(user_tensor)
                
                # Top-K 아이템 선택
                topk_scores, topk_indices = torch.topk(scores, k=k)
                
                # 결과 변환
                recommendations = [
                    (int(idx), float(score))
                    for idx, score in zip(topk_indices[0], topk_scores[0])
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
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'model_config': self.model_config,
                'is_trained': self.is_trained
            }, path)
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
            
        try:
            checkpoint = torch.load(path)
            self.config = checkpoint['config']
            self.model_config = checkpoint.get('model_config', self.model_config)
            self.is_trained = checkpoint['is_trained']
            
            if self.is_trained:
                config, train_data, _, _ = self._prepare_data(pd.DataFrame())
                self.model = RecBoleDSSM(config, train_data.dataset)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"모델 로드 완료: {path}")
                
        except Exception as e:
            self.is_trained = False
            logging.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 로드 실패: {str(e)}") 