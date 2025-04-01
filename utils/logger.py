import logging
import os
from datetime import datetime

def setup_logger(name='recommendation'):
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명 설정 (날짜 포함)
    log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d")}.log')
    
    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    
    return logger 