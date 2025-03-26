"""
DSSM(Deep Structured Semantic Model) 콘텐츠 기반 추천 모델

카테고리, 스타일, 임베딩 벡터 등의 아이템 특징을 활용하여
콘텐츠 기반 추천을 수행합니다.
"""

import os
import logging
import tempfile
import shutil
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import torch
from recbole.model.general_recommender import DSSM as RecBoleDSSM
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from models.base.base_recommender import BaseRecommender

class DSSMModel(BaseRecommender):
    """RecBole DSSM 모델 클래스"""
    
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
            'batch_size': self.config.get('batch_size', 256),
            'categorical_features': self.config.get('categorical_features', ['category', 'style']),
            'numerical_features': self.config.get('numerical_features', ['embedding_vector'])
        }
        
        self.model = None
        self.dataset = None
        self.config_obj = None
        self.user_ids = []
        self.item_ids = []
        self.is_trained = False
        
    def train(self, ratings: pd.DataFrame, items: pd.DataFrame) -> None:
        """
        모델 학습
        
        Args:
            ratings (pd.DataFrame): 'user_id', 'item_id', 'rating' 컬럼 필수
            items (pd.DataFrame): 아이템 특성 데이터
        """
        try:
            logging.info("DSSM 모델 학습 시작")
            
            # 임시 디렉토리 생성
            temp_dir = tempfile.mkdtemp()
            
            try:
                # RecBole 형식으로 데이터 변환
                self._prepare_data(ratings, items, temp_dir)
                
                # 모델 설정
                config_dict = {
                    'data_path': temp_dir,
                    'USER_ID_FIELD': 'user_id',
                    'ITEM_ID_FIELD': 'item_id',
                    'RATING_FIELD': 'rating',
                    'load_col': {
                        'inter': ['user_id', 'item_id', 'rating'],
                        'item': ['item_id'] + self.model_config['categorical_features'] + self.model_config['numerical_features']
                    },
                    'mlp_hidden_size': self.model_config['mlp_hidden_size'],
                    'dropout_prob': self.model_config['dropout_prob'],
                    'learning_rate': self.model_config['learning_rate'],
                    'epochs': self.model_config['n_epochs'],
                    'train_batch_size': self.model_config['batch_size'],
                    'eval_batch_size': 256,
                    'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'mode': 'full'},
                    'metrics': ['Recall', 'NDCG'],
                    'topk': [10],
                    'valid_metric': 'NDCG@10',
                }
                
                # RecBole 설정 객체 생성
                self.config_obj = Config(model='DSSM', dataset='ml-100k', config_dict=config_dict)
                
                # 데이터셋 생성
                self.dataset, _, _, _ = data_preparation(self.config_obj)
                
                # 모델 초기화
                self.model = RecBoleDSSM(self.config_obj, self.dataset)
                
                # 학습
                trainer = Trainer(self.config_obj, self.model)
                trainer.fit()
                
                # 사용자 및 아이템 ID 저장
                self.user_ids = self.dataset.field2id_token['user_id']
                self.item_ids = self.dataset.field2id_token['item_id']
                
                self.is_trained = True
                logging.info("DSSM 모델 학습 완료")
                
            finally:
                # 임시 디렉토리 삭제
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            self.is_trained = False
            logging.error(f"DSSM 모델 학습 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 학습 실패: {str(e)}")
    
    def _prepare_data(self, ratings: pd.DataFrame, items: pd.DataFrame, temp_dir: str) -> None:
        """
        RecBole 형식으로 데이터 준비
        
        Args:
            ratings (pd.DataFrame): 평점 데이터
            items (pd.DataFrame): 아이템 특성 데이터
            temp_dir (str): 임시 디렉토리 경로
        """
        # 필요한 컬럼만 선택
        ratings_df = ratings[['user_id', 'item_id', 'rating']].copy()
        
        # 아이템 특성 데이터 준비
        items_df = items[['item_id'] + self.model_config['categorical_features'] + self.model_config['numerical_features']].copy()
        
        # 파일로 저장
        os.makedirs(os.path.join(temp_dir, 'ml-100k'), exist_ok=True)
        ratings_df.to_csv(os.path.join(temp_dir, 'ml-100k', 'ml-100k.inter'), sep='\t', index=False)
        items_df.to_csv(os.path.join(temp_dir, 'ml-100k', 'ml-100k.item'), sep='\t', index=False)
    
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
    
    def get_similar_items(self, item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        특정 아이템과 유사한 아이템 Top-K 추출
        
        Args:
            item_id (int): 아이템 ID
            k (int): 유사 아이템 개수
            
        Returns:
            List[Tuple[int, float]]: (아이템 ID, 유사도 점수) 형태의 추천 리스트
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        try:
            # 내부 인덱스로 변환
            item_idx = self.dataset.token2id['item_id'].get(str(item_id))
            
            if item_idx is None:
                return []
            
            # 아이템 임베딩 추출
            with torch.no_grad():
                item_embedding = self.model.item_embedding(torch.LongTensor([item_idx]))
                
                # 모든 아이템 임베딩 추출
                all_items = torch.arange(len(self.item_ids))
                all_item_embeddings = self.model.item_embedding(all_items)
                
                # 코사인 유사도 계산
                similarity = torch.nn.functional.cosine_similarity(
                    item_embedding.repeat(len(all_items), 1),
                    all_item_embeddings
                )
                
                # Top-K 아이템 선택 (자기 자신 제외)
                mask = torch.ones_like(similarity, dtype=torch.bool)
                mask[item_idx] = False
                masked_similarity = similarity * mask.float()
                
                # Top-K 아이템 선택
                _, indices = torch.topk(masked_similarity, k)
                
                # 결과 변환
                similar_items = []
                for idx in indices:
                    similar_item_id = int(self.item_ids[idx])
                    score = float(similarity[idx].item())
                    similar_items.append((similar_item_id, score))
                
                return similar_items
                
        except Exception as e:
            logging.error(f"유사 아이템 검색 중 오류 발생: {str(e)}")
            return []
    
    def save_model(self, path: str) -> None:
        """
        모델 저장
        
        Args:
            path (str): 저장 경로
        """
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        try:
            logging.info(f"DSSM 모델 저장 시작: {path}")
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
            logging.info("DSSM 모델 저장 완료")
            
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
            logging.info(f"DSSM 모델 로드 시작: {path}")
            
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
                    'load_col': {
                        'inter': ['user_id', 'item_id', 'rating'],
                        'item': ['item_id'] + self.model_config['categorical_features'] + self.model_config['numerical_features']
                    },
                    'mlp_hidden_size': self.model_config['mlp_hidden_size'],
                }
                
                # 빈 파일 생성
                os.makedirs(os.path.join(temp_dir, 'ml-100k'), exist_ok=True)
                with open(os.path.join(temp_dir, 'ml-100k', 'ml-100k.inter'), 'w') as f:
                    f.write('user_id\titem_id\trating\n')
                with open(os.path.join(temp_dir, 'ml-100k', 'ml-100k.item'), 'w') as f:
                    header = ['item_id'] + self.model_config['categorical_features'] + self.model_config['numerical_features']
                    f.write('\t'.join(header) + '\n')
                
                # 설정 및 데이터셋 생성
                self.config_obj = Config(model='DSSM', dataset='ml-100k', config_dict=config_dict)
                self.dataset, _, _, _ = data_preparation(self.config_obj)
                
                # 모델 초기화 및 가중치 로드
                self.model = RecBoleDSSM(self.config_obj, self.dataset)
                self.model.load_state_dict(model_state['model_state_dict'])
                self.model.eval()
                
            finally:
                # 임시 디렉토리 삭제
                shutil.rmtree(temp_dir)
            
            logging.info("DSSM 모델 로드 완료")
            
        except Exception as e:
            self.is_trained = False
            logging.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise RuntimeError(f"모델 로드 실패: {str(e)}") 