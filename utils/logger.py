"""
로깅 설정 및 유틸리티 모듈
"""

import logging
import os
from datetime import datetime

def setup_logger(name='recommendation'):
    """
    로거를 설정하고 반환합니다.
    
    Args:
        name (str): 로거 이름
        
    Returns:
        logger: 설정된 로거 객체
    """
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 있는 경우 중복 설정 방지
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 생성
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 로거에 핸들러 추가
    logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (logs 디렉토리)
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger