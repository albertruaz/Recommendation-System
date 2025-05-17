"""
추천 시스템 메인 실행 스크립트
"""

import os
import json
import pandas as pd
from run.run_als import RunALS
from run.run_als2 import RunALSTuning
from utils.logger import setup_logger

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
        for max_iter in [10, 30, 50]:
            for reg_param in [0.001, 0.01, 0.1]:
                for rank in [10, 20, 30]:
                    als2_runner = RunALSTuning(max_iter, reg_param, rank)
                    result = als2_runner.run()
                    logger.info(f"max_iter: {max_iter}, reg_param: {reg_param}, rank: {rank}, result: {result}")

        return result
    except Exception as e:
        logger.error(f"메인 실행 중 오류 발생: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()
