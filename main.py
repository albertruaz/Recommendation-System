"""
추천 시스템 메인 실행 스크립트
"""

import os
import json
import pandas as pd
from run.run_als import RunALS
from run.run_recent_product import RunRecentProduct
from utils.logger import setup_logger
from utils.logger import log_als_model_results, log_recent_product_model_results

# 로거 설정
logger = setup_logger('main')

def main():
    """
    메인 실행 함수 - 여러 run 모듈을 조합하여 실행합니다.
    """
    logger.info("추천 시스템 메인 실행 시작")
    
    try:
        # ALS 모델 추천 생성
        # logger.info("ALS 모듈 실행")
        # als_runner = RunALS()
        # result = als_runner.run()
        # log_als_model_results(result)   
        
        logger.info("최근 본 상품과 유사한 상품 추천 실행")
        recent_product_runner = RunRecentProduct()
        result = recent_product_runner.run()
        log_recent_product_model_results(result)
        
        return result
    except Exception as e:
        logger.error(f"메인 실행 중 오류 발생: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()
