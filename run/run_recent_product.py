"""
최근 장바구니 상품 기반 추천 시스템 실행 클래스
"""

import os
import pandas as pd
from datetime import datetime
from model_recent_product.recent_product_model import RecentProductModel
from utils.logger import setup_logger
from utils.recommendation_utils import save_recommendations

class RunRecentProduct:
    def __init__(self):
        """
        최근 장바구니 상품 기반 추천 실행 클래스 초기화
        """
        # 로거 설정
        self.logger = setup_logger('recent_product')
        
        # 기본 설정
        self.days = 30  # 최근 30일 데이터 사용
        self.top_n = 100  # 사용자당 100개 추천
        self.output_dir = 'output'  # 결과 저장 경로
        
        # 모델 설정 추가 (als_config와 유사한 형태로)
        self.recent_product_config = {
            "model_type": "recent_product",
            "params": {
                "days": self.days,
                "top_n": self.top_n
            }
        }
        
    def run(self):
        """
        최근 장바구니 상품 기반 추천 생성 실행
        
        Returns:
            추천 결과 정보를 담고 있는 딕셔너리
        """
        self.logger.info("최근 장바구니 상품 기반 추천 시작")
        
        try:
            # 고유한 실행 ID 생성 (타임스탬프 기반)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"recent_product_{timestamp}"
            output_dir_with_id = os.path.join(self.output_dir, run_id)
            os.makedirs(output_dir_with_id, exist_ok=True)
            
            # 모델 초기화
            model = RecentProductModel(top_n=self.top_n)
            
            # 추천 생성
            self.logger.info(f"최근 {self.days}일간의 장바구니 데이터로 추천 생성 중...")
            recommendations_df = model.generate_recommendations(days=self.days)
            
            # 추천 결과 분석 및 로깅
            if recommendations_df.empty:
                self.logger.warning("생성된 추천 결과가 없습니다.")
            else:
                user_count = recommendations_df['member_id'].nunique()
                product_count = recommendations_df['product_id'].nunique()
                avg_recommendations = len(recommendations_df) / user_count if user_count > 0 else 0
                
                self.logger.info(f"추천 결과 요약:")
                self.logger.info(f"- 추천 생성된 사용자 수: {user_count}명")
                self.logger.info(f"- 추천된 총 상품 수: {product_count}개")
                self.logger.info(f"- 사용자당 평균 추천 수: {avg_recommendations:.2f}개")
                
                # 평균 유사도 점수 계산
                avg_score = recommendations_df['score'].mean() if 'score' in recommendations_df.columns else 0
                self.logger.info(f"- 평균 유사도 점수: {avg_score:.4f}")
                
                # 추천 결과 저장
                output_file = os.path.join(output_dir_with_id, f"{run_id}.csv")
                recommendations_df.to_csv(output_file, index=False)
                self.logger.info(f"추천 결과가 {output_file}에 저장되었습니다.")
                
                # 다른 형식으로도 저장 (utils 사용)
                save_recommendations(recommendations_df, output_dir=output_dir_with_id)
                
            # 결과 반환 (run_id와 config 추가)
            result = {
                "run_id": run_id,
                "recommendations": recommendations_df,
                "user_count": recommendations_df['member_id'].nunique() if not recommendations_df.empty else 0,
                "product_count": recommendations_df['product_id'].nunique() if not recommendations_df.empty else 0,
                "total_recommendations": len(recommendations_df),
                "output_dir": output_dir_with_id,
                "config": self.recent_product_config
            }
            
            self.logger.info("최근 장바구니 상품 기반 추천 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"추천 생성 중 오류 발생: {str(e)}", exc_info=True)
            raise

# 직접 실행할 경우의 예시
if __name__ == "__main__":
    try:
        runner = RunRecentProduct()
        result = runner.run()
        
        if 'recommendations' in result and not result['recommendations'].empty:
            print(f"추천 결과: {result['total_recommendations']}개")
            print(f"추천 대상 사용자 수: {result['user_count']}명")
            print(f"추천된 상품 수: {result['product_count']}개")
            print(f"결과 저장 경로: {result['output_dir']}")
        else:
            print("추천 결과가 없습니다.")
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}") 