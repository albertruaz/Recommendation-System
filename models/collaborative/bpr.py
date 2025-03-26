"""
BPR(Bayesian Personalized Ranking) 협업 필터링 모델
"""

import os
import logging
import tempfile
import shutil
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import torch
from recbole.model.general_recommender import BPR as RecBoleBPR
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from models.base.base_recommender import BaseRecommender

class BPRModel(BaseRecommender):
    """RecBole BPR 모델 클래스"""
    
    def __init__(self, config_dict=None):
        """
        Args:
            config_dict (dict, optional): 모델 설정
        """
        super().__init__()
        self.config = config_dict or {}
        self.model_config = {
            'embedding_size': self.config.get('embedding_size', 64),
            'learning_rate': self.config.get('learning_rate', 0.01),
            'n_epochs': self.config.get('n_epochs', 100),
            'batch_size': self.config.get('batch_size', 2048),
            'neg_sampling': self.config.get('neg_sampling', {
                'strategy': 'full',
                'distribution': 'uniform'
            })
        }
        
        self.model = None
        self.dataset = None
        self.config_obj = None
        self.user_ids = []
        self.item_ids = []
        self.is_trained = False
        
    def train(self, ratings: pd.DataFrame) -> None:
        """
        모델 학습
        
        Args:
            ratings (pd.DataFrame): 'user_id', 'item_id', 'rating' 컬럼 필수
        """
        try:
            logging.info("BPR 모델 학습 시작")
            
            # 임시 디렉토리 생성
            temp_dir = tempfile.mkdtemp()
            
            try:
                # RecBole 형식으로 데이터 변환
                self._prepare_data(ratings, temp_dir)
                
                # 모델 설정
                config_dict = {
                    'data_path': temp_dir,
                    'USER_ID_FIELD': 'user_id',
                    'ITEM_ID_FIELD': 'item_id',
                    'RATING_FIELD': 'rating',
                    'load_col': {'inter': ['user_id', 'item_id', 'rating']},
                    'neg_sampling': self.model_config['neg_sampling'],
                    'embedding_size': self.model_config['embedding_size'],
                    'learning_rate': self.model_config['learning_rate'],
                    'epochs': self.model_config['n_epochs'],
                    'train_batch_size': self.model_config['batch_size'],
                    'eval_batch_size': 4096,
                    'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'mode': 'full'},
                    'metrics': ['Recall', 'NDCG'],
                    'topk': [10],
                    'valid_metric': 'NDCG@10',
                    'train_neg_sample_args': None,
                }
                
                # RecBole 설정 객체 생성
                self.config_obj = Config(model='BPR', dataset='ml-100k', config_dict=config_dict)
                
                # 데이터셋 생성
                self.dataset, _, _, _ = data_preparation(self.config_obj)
                
                # 모델 초기화
                self.model = RecBoleBPR(self.config_obj, self.dataset)
                
                # 학습
                trainer = Trainer(self.config_obj, self.model)
                trainer.fit()
                
                # 사용자 및 아이템 ID 저장
                self.user_ids = self.dataset.field2id_token['user_id']
                self.item_ids = self.dataset.field2id_token['item_id']
                
                self.is_trained = True
                logging.info("BPR 모델 학습 완료")
                
            finally:
                # 임시 디렉토리 삭제
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            self.is_trained = False
            logging.error(f"BPR 모델 학습 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 학습 실패: {str(e)}")
    
    def _prepare_data(self, ratings: pd.DataFrame, temp_dir: str) -> None:
        """
        RecBole 형식으로 데이터 준비
        
        Args:
            ratings (pd.DataFrame): 평점 데이터
            temp_dir (str): 임시 디렉토리 경로
        """
        # 필요한 컬럼만 선택
        df = ratings[['user_id', 'item_id', 'rating']].copy()
        
        # 파일로 저장
        os.makedirs(os.path.join(temp_dir, 'ml-100k'), exist_ok=True)
        df.to_csv(os.path.join(temp_dir, 'ml-100k', 'ml-100k.inter'), sep='\t', index=False)
    
    def get_score(self, user_id: int, item_id: int) -> float:
        """
        특정 사용자-아이템 쌍에 대한 점수 계산
        
        Args:
            user_id (int): 사용자 ID
            item_id (int): 아이템 ID
            
        Returns:
            float: 예측 점수
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        try:
            # 내부 인덱스로 변환
            user_idx = self.dataset.token2id['user_id'].get(str(user_id))
            item_idx = self.dataset.token2id['item_id'].get(str(item_id))
            
            if user_idx is None or item_idx is None:
                return 0.0
            
            # 텐서로 변환
            user_tensor = torch.LongTensor([user_idx])
            item_tensor = torch.LongTensor([item_idx])
            
            # 점수 계산
            with torch.no_grad():
                score = self.model.predict(user_tensor, item_tensor).item()
            
            return float(score)
            
        except Exception as e:
            logging.error(f"점수 계산 중 오류 발생: {str(e)}")
            return 0.0
    
    def get_recommendations(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        특정 사용자에 대한 추천 아이템 Top-K 추출
        
        Args:
            user_id (int): 사용자 ID
            k (int): 추천할 아이템 개수
            
        Returns:
            List[Tuple[int, float]]: (아이템 ID, 점수) 형태의 추천 리스트
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        try:
            # 내부 인덱스로 변환
            user_idx = self.dataset.token2id['user_id'].get(str(user_id))
            
            if user_idx is None:
                return []
            
            # 모든 아이템에 대한 점수 계산
            user_tensor = torch.LongTensor([user_idx])
            with torch.no_grad():
                # 모든 아이템에 대한 점수 계산
                all_items = torch.arange(len(self.item_ids))
                scores = self.model.predict(user_tensor.repeat(len(all_items)), all_items)
                
                # 상호작용한 아이템 마스킹 (학습 데이터에 있는 아이템 제외)
                mask = torch.ones_like(scores, dtype=torch.bool)
                for item_idx in self.dataset.user_item_feat_interaction['train'][user_idx]:
                    mask[item_idx] = False
                
                # 마스킹된 점수
                masked_scores = scores * mask.float()
                
                # Top-K 아이템 선택
                _, indices = torch.topk(masked_scores, k)
                
                # 결과 변환
                recommendations = []
                for idx in indices:
                    item_id = int(self.item_ids[idx])
                    score = float(scores[idx].item())
                    recommendations.append((item_id, score))
                
                return recommendations
                
        except Exception as e:
            logging.error(f"추천 생성 중 오류 발생: {str(e)}")
            return []

    def get_all_recommendations(self, k: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """
        모든 사용자에 대한 추천 아이템 Top-K 추출
        
        Args:
            k (int): 추천할 아이템 개수
            
        Returns:
            Dict[int, List[Tuple[int, float]]]: {user_id: [(item_id, 점수), ...]} 형태의 딕셔너리
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        try:
            all_recommendations = {}
            
            # 모든 사용자에 대해 추천 생성
            for user_id in self.user_ids:
                user_id_int = int(user_id)
                recommendations = self.get_recommendations(user_id_int, k=k)
                all_recommendations[user_id_int] = recommendations
            
            return all_recommendations
            
        except Exception as e:
            logging.error(f"전체 추천 생성 중 오류 발생: {str(e)}")
            return {}
    
    def save_model(self, path: str) -> None:
        """
        모델 저장
        
        Args:
            path (str): 저장 경로
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        try:
            logging.info(f"BPR 모델 저장 시작: {path}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 모델 상태 저장
            model_state = {
                'config': self.config,
                'model_config': self.model_config,
                'model_state_dict': self.model.state_dict(),
                'user_ids': self.user_ids,
                'item_ids': self.item_ids,
                'is_trained': self.is_trained
            }
            
            torch.save(model_state, path)
            logging.info("BPR 모델 저장 완료")
            
        except Exception as e:
            logging.error(f"모델 저장 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 저장 실패: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """
        모델 로드
        
        Args:
            path (str): 모델 파일 경로
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        
        try:
            logging.info(f"BPR 모델 로드 시작: {path}")
            
            # 모델 상태 로드
            model_state = torch.load(path)
            
            self.config = model_state['config']
            self.model_config = model_state['model_config']
            self.user_ids = model_state['user_ids']
            self.item_ids = model_state['item_ids']
            self.is_trained = model_state['is_trained']
            
            # 임시 설정 및 데이터셋 생성
            temp_dir = tempfile.mkdtemp()
            try:
                # 설정 객체 생성
                config_dict = {
                    'data_path': temp_dir,
                    'USER_ID_FIELD': 'user_id',
                    'ITEM_ID_FIELD': 'item_id',
                    'RATING_FIELD': 'rating',
                    'load_col': {'inter': ['user_id', 'item_id', 'rating']},
                    'embedding_size': self.model_config['embedding_size'],
                }
                
                # 빈 파일 생성
                os.makedirs(os.path.join(temp_dir, 'ml-100k'), exist_ok=True)
                with open(os.path.join(temp_dir, 'ml-100k', 'ml-100k.inter'), 'w') as f:
                    f.write('user_id\titem_id\trating\n')
                
                # 설정 및 데이터셋 생성
                self.config_obj = Config(model='BPR', dataset='ml-100k', config_dict=config_dict)
                self.dataset, _, _, _ = data_preparation(self.config_obj)
                
                # 모델 초기화 및 가중치 로드
                self.model = RecBoleBPR(self.config_obj, self.dataset)
                self.model.load_state_dict(model_state['model_state_dict'])
                self.model.eval()
                
            finally:
                # 임시 디렉토리 삭제
                shutil.rmtree(temp_dir)
            
            logging.info("BPR 모델 로드 완료")
            
        except Exception as e:
            self.is_trained = False
            logging.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 로드 실패: {str(e)}") 