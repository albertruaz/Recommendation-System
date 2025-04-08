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
    
    def _get_interaction_weight(self, row: pd.Series) -> float:
        """상호작용 타입에 따른 가중치 반환"""
        interaction_type = row["interaction_type"]
        view_type = f"view_type_{int(row['view_type'])}" if pd.notna(row["view_type"]) else None
        
        if view_type and view_type in CONFIG["interaction_weights"]:
            return CONFIG["interaction_weights"][view_type]
        elif interaction_type in CONFIG["interaction_weights"]:
            return CONFIG["interaction_weights"][interaction_type]
        return 1.0
    
    def _prepare_matrices(self, interactions_df: pd.DataFrame) -> None:
        """상호작용 데이터를 행렬 형태로 변환"""
        # 인덱스 매핑 생성
        self._prepare_indices(interactions_df)
        
        # 가중치 계산
        interactions_df["weight"] = interactions_df.apply(self._get_interaction_weight, axis=1)
        
        # 사용자-상품별로 가중치 합산
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
        self.data = MatrixMarket(als_opt)
        self.data.create()
        
        # 임시 파일 삭제
        if os.path.exists(temp_mm_file):
            os.remove(temp_mm_file)
        
        # 행렬 정보 로깅
        self.logger.info(f"\n행렬 변환 결과:")
        self.logger.info(f"- 행렬 크기: {len(self.user2idx)}x{len(self.item2idx)}")
        self.logger.info(f"- 비영요소 수: {len(aggregated_df)}")
        self.logger.info(f"- 밀도: {len(aggregated_df) / (len(self.user2idx) * len(self.item2idx)):.4%}")
        
        # 샘플 데이터 출력 (처음 10x10 행렬)
        sample_size = min(10, sparse_matrix.shape[0], sparse_matrix.shape[1])
        # COO를 CSR로 변환하여 인덱싱 가능하게 만듦
        sparse_matrix_csr = sparse_matrix.tocsr()
        sample_matrix = sparse_matrix_csr[:sample_size, :sample_size].toarray()
        
        # 행과 열 인덱스 준비
        row_indices = [f"User {self.idx2user[i]}" for i in range(sample_size)]
        col_indices = [f"Item {self.idx2item[i]}" for i in range(sample_size)]
        
        # 행렬 출력 준비
        matrix_str = "\n행렬 샘플 (10x10):\n"
        
        # 열 헤더 추가
        max_user_len = max(len(str(u)) for u in row_indices)
        matrix_str += " " * (max_user_len + 2)
        for col in col_indices:
            matrix_str += f"{col:>10} "
        matrix_str += "\n"
        
        # 데이터 행 추가
        for i, user in enumerate(row_indices):
            matrix_str += f"{user:<{max_user_len}} |"
            for j in range(sample_size):
                matrix_str += f"{sample_matrix[i,j]:10.3f} "
            matrix_str += "\n"
        
        self.logger.info(matrix_str)
    
    def train(self, interactions_df: pd.DataFrame) -> None:
        """모델 학습"""
        try:
            self.logger.info("Buffalo ALS 모델 학습 시작")
            
            # 데이터 준비
            self._prepare_matrices(interactions_df)
            
            # ALSOption을 사용하여 기본 옵션 가져오기
            als_opt = ALSOption().get_default_option()
            
            # 사용자 지정 파라미터로 업데이트 (타입 변환 추가)
            als_opt.d = int(self.rank)  # 잠재 요인 차원 수
            als_opt.num_iters = int(self.max_iter)  # 반복 횟수
            als_opt.reg_u = float(self.reg_param)  # 사용자 정규화 계수
            als_opt.reg_i = float(self.reg_param)  # 아이템 정규화 계수
            als_opt.alpha = float(self.alpha)  # 신뢰도 가중치
            
            # 추가 설정
            als_opt.num_workers = 4  # 스레드 수
            als_opt.validation = aux.Option({"topk": 10})  # 검증 설정
            als_opt.compute_loss_on_training = True  # 학습 중 손실 계산
            als_opt.save_factors = True  # 모델 저장
            als_opt.model_path = "output/als_model"  # 모델 저장 경로
            
            # 모델 생성 및 학습
            self.model = ALS(als_opt)
            self.model.initialize()  # 초기화 필요
            self.model.train(self.data)
            
            # 학습된 잠재 요인 저장
            self.user_factors = self.model.P
            self.item_factors = self.model.Q
            
            self.logger.info("모델 학습 완료")
            
            # 학습 결과 분석
            self.logger.info("\n=== 학습 결과 분석 ===")
            
            # 원본 데이터에서 실제 가중치 정보 추출
            analysis_df = interactions_df.copy()
            analysis_df['actual_weight'] = analysis_df.apply(self._get_interaction_weight, axis=1)
            
            # 예측값 계산
            predictions = []
            for _, row in analysis_df.iterrows():
                user_idx = self.user2idx[row['member_id']]
                item_idx = self.item2idx[row['product_id']]
                pred_score = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                predictions.append({
                    'user_id': row['member_id'],
                    'item_id': row['product_id'],
                    'interaction_type': row['interaction_type'],
                    'view_type': row['view_type'] if 'view_type' in row else None,
                    'actual_weight': row['actual_weight'],
                    'predicted_score': pred_score
                })
            
            # 결과를 DataFrame으로 변환하고 정렬
            result_df = pd.DataFrame(predictions)
            result_df = result_df.sort_values('actual_weight', ascending=False)
            
            # 출력 디렉토리 생성
            os.makedirs('output', exist_ok=True)
            
            # CSV 파일로 저장
            output_path = 'output/weight_analysis.csv'
            result_df.to_csv(output_path, index=False)
            
            # 상위 100개 결과 로깅
            self.logger.info("\n=== 상위 100개 가중치 비교 ===")
            self.logger.info("User ID | Item ID | Type | View Type | Actual Weight | Predicted Score")
            self.logger.info("-" * 75)
            
            for _, row in result_df.head(100).iterrows():
                view_type_str = f"type_{int(row['view_type'])}" if pd.notna(row['view_type']) else "N/A"
                view_type_display = view_type_str if row['interaction_type'] == 'view' else "-"
                
                self.logger.info(
                    f"{row['user_id']:7d} | "
                    f"{row['item_id']:7d} | "
                    f"{row['interaction_type']:<6s} | "
                    f"{view_type_display:^9s} | "
                    f"{row['actual_weight']:13.2f} | "
                    f"{row['predicted_score']:14.2f}"
                )
            
            # 상호작용 타입별 통계
            self.logger.info("\n=== 상호작용 타입별 통계 ===")
            type_stats = result_df.groupby('interaction_type').agg({
                'actual_weight': ['mean', 'min', 'max', 'count'],
                'predicted_score': ['mean', 'min', 'max']
            }).round(3)
            
            self.logger.info("\n" + str(type_stats))
            self.logger.info(f"\n전체 분석 결과가 {output_path}에 저장되었습니다.")
            
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