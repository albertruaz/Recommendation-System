"""
Buffalo ALS를 사용한 명시적 피드백 기반 추천 시스템

이 모듈은 Buffalo ALS를 사용하여 명시적 피드백 기반의 사용자-상품 추천을 생성하는 클래스를 제공합니다.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
from buffalo.algo.als import ALS
from buffalo.data.mm import MatrixMarket, MatrixMarketOptions
from buffalo.misc import aux
from .base_als import BaseALS, CONFIG
from buffalo.algo.options import ALSOption

# numpy 출력 설정
np.set_printoptions(threshold=np.inf, precision=3, suppress=True)

class BuffaloALS(BaseALS):
    """Buffalo ALS 기반 추천 시스템 클래스"""
    
    def __init__(self, max_iter: int = 15, reg_param: float = 0.1,
                 rank: int = 100, alpha: float = 40):
        """
        Args:
            max_iter (int): 최대 반복 횟수
            reg_param (float): 정규화 파라미터
            rank (int): 잠재 요인 개수
            alpha (float): 신뢰도 가중치 스케일링 파라미터
        """
        super().__init__(max_iter, reg_param, rank)
        self.alpha = alpha
        self.model = None
        self.data = None
    
    def prepare_matrices(self, interactions_df: pd.DataFrame):
        """상호작용 데이터를 Buffalo 라이브러리용 행렬로 변환
        
        부모 클래스의 prepare_matrices를 오버라이드하여 
        Buffalo 라이브러리에 맞는 MatrixMarket 객체를 생성합니다.
        
        Args:
            interactions_df (pd.DataFrame): 상호작용 데이터프레임
            
        Returns:
            MatrixMarket: Buffalo 라이브러리용 행렬 객체
        """
        # 기본 행렬 변환을 위해 부모 메서드 호출 (결과는 사용하지 않음)
        _ = super().prepare_matrices(interactions_df)
        
        # BuffaloALS 전용 구현: MatrixMarket 생성
        # 가중치 계산 (이미 부모 클래스에서 계산됨)
        aggregated_df = interactions_df.groupby(
            ["member_id", "product_id"]
        )["weight"].max().reset_index()
        
        # 임시 파일 생성
        temp_mm_file = "temp_matrix.mtx"
        
        # 인덱스 매핑 적용 및 행렬 생성
        user_indices = aggregated_df['member_id'].map(self.user2idx).values
        item_indices = aggregated_df['product_id'].map(self.item2idx).values
        weights = aggregated_df['weight'].values
        
        # COO 형식으로 변환
        sparse_matrix = sp.coo_matrix(
            (weights, (user_indices, item_indices)),
            shape=(len(self.user2idx), len(self.item2idx))
        )
        
        # MatrixMarket 형식으로 저장
        scipy.io.mmwrite(temp_mm_file, sparse_matrix)
        
        # 옵션 객체 생성
        als_opt = ALSOption().get_default_option()

        # 주요 하이퍼파라미터 설정
        als_opt.d = int(self.rank)
        als_opt.num_iters = int(self.max_iter)
        als_opt.reg_u = float(self.reg_param)
        als_opt.reg_i = float(self.reg_param)
        als_opt.alpha = float(self.alpha)

        # 조기 종료 및 저장 옵션
        als_opt.early_stopping_rounds = 3
        als_opt.save_best = True  # 이거 안 넣으면 RuntimeError 발생함!

        # 기타 설정
        als_opt.num_workers = 4
        als_opt.validation = aux.Option({"topk": 10})
        als_opt.compute_loss_on_training = True
        als_opt.save_factors = True
        als_opt.model_path = "output/als_model"
        
        # MatrixMarket 객체 생성
        data = MatrixMarket(als_opt)
        data.create()
        
        # 임시 파일 삭제
        if os.path.exists(temp_mm_file):
            os.remove(temp_mm_file)
        
        return data
    
    def train(self, interactions_df, matrix_data) -> None:
        """모델 학습
        
        Args:
            interactions_df: 상호작용 데이터프레임
            matrix_data: 미리 준비된 행렬 데이터 (MatrixMarket 객체)
        """
        try:
            self.logger.info("Buffalo ALS 모델 학습 시작")
            
            # 데이터 설정 - 항상 준비된 행렬 데이터 사용
            self.data = matrix_data
            
            # 모델 생성 및 학습
            als_opt = ALSOption().get_default_option()
            
            # 사용자 지정 파라미터로 업데이트 (타입 변환 추가)
            als_opt.d = int(self.rank)  # 잠재 요인 차원 수
            als_opt.num_iters = int(self.max_iter)  # 반복 횟수
            als_opt.reg_u = float(self.reg_param)  # 사용자 정규화 계수
            als_opt.reg_i = float(self.reg_param)  # 아이템 정규화 계수
            als_opt.alpha = float(self.alpha)  # 신뢰도 가중치
            
            # 기타 설정
            als_opt.num_workers = 4  # 스레드 수
            als_opt.validation = aux.Option({"topk": 10})  # 검증 설정
            als_opt.compute_loss_on_training = True  # 학습 중 손실 계산
            
            self.model = ALS(als_opt)
            self.model.initialize()  # 초기화 필요
            self.model.train(self.data)
            
            # 학습된 잠재 요인 저장
            self.user_factors = self.model.P
            self.item_factors = self.model.Q
            
            self.logger.info("모델 학습 완료")
            
            # 학습 결과 분석 및 로깅
            self._log_training_results(interactions_df)
            
        except Exception as e:
            self.logger.error(f"모델 학습 오류: {str(e)}")
            raise
    
    def generate_recommendations(self, top_n: int = 300) -> pd.DataFrame:
        """전체 사용자에 대한 추천 생성
        
        Args:
            top_n (int): 각 사용자당 추천할 상품 수
            
        Returns:
            pd.DataFrame: 추천 결과 데이터프레임
        """
        if self.model is None:
            raise Exception("모델이 학습되지 않았습니다. train() 메서드를 먼저 실행하세요.")
        
        try:
            results = []
            
            # 각 사용자에 대해 추천 생성
            for user_idx, user_id in enumerate(self.idx2user):
                # 예측 점수 계산
                scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
                
                # 상위 N개 아이템 선택
                top_item_indices = np.argsort(-scores)[:top_n]
                top_scores = scores[top_item_indices]
                
                # 결과 저장
                user_results = pd.DataFrame({
                    'member_id': user_id,
                    'product_id': [self.idx2item[idx] for idx in top_item_indices],
                    'predicted_rating': top_scores
                })
                results.append(user_results)
            
            # 모든 결과 합치기
            recommendations_df = pd.concat(results, ignore_index=True)
            return recommendations_df
            
        except Exception as e:
            raise Exception(f"추천 생성 오류: {str(e)}")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        super().cleanup()
        self.model = None
        self.data = None
        
        # 임시 파일 삭제
        try:
            if os.path.exists('temp/ratings.mtx'):
                os.remove('temp/ratings.mtx')
            if os.path.exists('temp'):
                os.rmdir('temp')
        except Exception as e:
            self.logger.warning(f"임시 파일 삭제 중 오류 발생: {str(e)}") 