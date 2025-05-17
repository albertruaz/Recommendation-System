import logging
import os
import json
from datetime import datetime
import uuid

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

def overall_log(train_result, test_result):
    """
    ALS 실행 결과를 JSON 형식으로 기록
    
    Args:
        train_result: 학습 데이터 평가 결과
        test_result: 테스트 데이터 평가 결과
    """
    # 로그 파일 경로
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'overall_als.log')
    
    # name 생성 (현재 날짜/시간 + UUID)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # UUID의 첫 8자만 사용
    run_id = f"{timestamp}_{unique_id}"
    
    # ALS 설정 불러오기
    config_path = 'config/als_config.json'
    try:
        with open(config_path, 'r') as f:
            als_config = json.load(f)
    except Exception as e:
        als_config = {"error": f"설정 파일 로드 실패: {str(e)}"}
    
    # 로그 데이터 구성
    log_data = {
        "name": run_id,
        "check": train_result is not None,  # 학습 결과가 있으면 성공으로 간주
        "als_config": als_config,
        "train_result": train_result,
        "test_result": test_result
    }
    
    # 파일에 로그 추가
    try:
        # 기존 로그 읽기
        existing_logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # 각 줄을 JSON 객체로 파싱
                    for line in content.split('\n'):
                        if line.strip():  # 빈 줄 무시
                            existing_logs.append(json.loads(line))
        
        # 새 로그 추가
        existing_logs.append(log_data)
        
        # 파일에 저장 - 들여쓰기와 줄바꿈 포맷으로 저장
        with open(log_file, 'w', encoding='utf-8') as f:
            for log_entry in existing_logs:
                # 각 로그 항목을 들여쓰기된 JSON으로 변환하고 줄바꿈 추가
                formatted_json = json.dumps(log_entry, ensure_ascii=False, indent=2)
                f.write(formatted_json + '\n\n')  # 로그 항목 사이에 빈 줄 추가
        
        # 로깅 성공 메시지
        print(f"실행 로그가 {log_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"로그 저장 중 오류 발생: {str(e)}")