"""
로깅 설정 유틸리티 함수
"""

import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO) -> None:
    """
    로깅 설정을 초기화합니다.
    
    Args:
        log_dir (str): 로그 파일을 저장할 디렉토리
        log_level (int): 로깅 레벨
    """
    # 로그 디렉토리가 없으면 생성
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 로그 파일 경로
    log_file = os.path.join(log_dir, 'recommendation.log')
    
    # 로깅 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 파일 핸들러 설정 (최대 10MB, 백업 5개)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    root_logger.handlers = []
    
    # 새 핸들러 추가
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler) 