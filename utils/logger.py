import logging
import os
from datetime import datetime

# 로그 파일이 이미 생성되었는지 추적하는 변수
_log_files_created = {}

def setup_logger(name='recommendation'):
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명 설정 (날짜만 포함)
    log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d")}.log')
    
    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 파일 모드 결정 (첫 실행 시에만 덮어쓰기, 이후에는 추가)
    file_mode = 'w' if log_file not in _log_files_created else 'a'
    
    # 이 파일이 생성되었음을 기록
    _log_files_created[log_file] = True
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정 - 간소화된 포맷 사용
    file_formatter = logging.Formatter('%(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger