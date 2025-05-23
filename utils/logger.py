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

def log_als_model_results(result):
    """
    모델 실행 결과를 로깅하는 함수
    
    Args:
        run_id: 실행 ID
        train_result: 학습 데이터 평가 결과 (ALS의 경우)
        test_result: 테스트 데이터 평가 결과 (ALS의 경우)
        model_config: 모델 설정 정보

    """
    run_id = result['run_id']
    recommendations = result['recommendations']
    train_result = result['train_result']
    test_result = result['test_result']
    model_config = result['config']

    model_type = "unknown"
    if model_config and "model_type" in model_config:
        model_type = model_config["model_type"]
    
    # 로깅 결과 저장
    if train_result is not None and model_type == "pyspark_als":
        log_train_test_metrics(train_result, test_result)
    
    # overall_log 함수 호출 (모든 모델 타입에 공통)
    overall_log(run_id, train_result, test_result, model_config)

def log_similars_model_results(result):
    """
    장바구니 상품 유사도 기반 추천 실행 결과를 로깅하는 함수
    """
    pass

def log_train_test_metrics(train_result, test_result):
    """
    학습 및 테스트 메트릭을 로깅하는 함수
    
    Args:
        train_result: 학습 데이터 평가 결과
        test_result: 테스트 데이터 평가 결과
    """
    logger = logging.getLogger("recommendation")
    
    # 학습 결과 정보 출력 (있는 경우)
    if train_result is not None:
        logger.info(f"학습 데이터 결과:")
        logger.info(f"- MAE: {train_result['mae']:.4f}")
        logger.info(f"- RMSE: {train_result['rmse']:.4f}")
        logger.info(f"- 샘플 수: {train_result['samples']}")
    
    # 테스트 결과 정보 출력 (있는 경우)
    if test_result is not None:
        logger.info(f"테스트 데이터 결과:")
        logger.info(f"- MAE: {test_result['mae']:.4f}")
        logger.info(f"- RMSE: {test_result['rmse']:.4f}")
        logger.info(f"- 샘플 수: {test_result['samples']}")

def overall_log(run_id, train_result, test_result, model_config=None):
    """
    모델 실행 결과를 JSON 형식으로 기록
    
    Args:
        run_id: 실행 ID (timestamp 형식)
        train_result: 학습 데이터 평가 결과
        test_result: 테스트 데이터 평가 결과
        model_config: 모델 설정 정보
    """
    # 모델 타입 확인 (기본값: als)
    model_type = "als"
    if model_config and "model_type" in model_config:
        model_type = model_config["model_type"]
    
    # 로그 파일 경로
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    config_log_file = os.path.join(log_dir, f'overall_{model_type}_config.log')
    result_log_file = os.path.join(log_dir, f'overall_{model_type}_result.log')
    
    # 모델 설정 불러오기 (전달받지 않은 경우)
    if model_config is None:
        config_path = f'config/{model_type}_config.json'
        try:
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        except Exception as e:
            model_config = {"error": f"설정 파일 로드 실패: {str(e)}"}
    
    # 로그 데이터 구성 - 설정 로그
    config_log_data = {
        "name": run_id,
        "model_config": model_config
    }
    
    # 로그 데이터 구성 - 결과 로그
    result_log_data = {
        "name": run_id,
        "check": train_result is not None if train_result is not None else True,  # 결과가 없으면 성공으로 간주
        "train_result": train_result,
        "test_result": test_result
    }
    
    # 설정 로그 파일에 로그 추가
    try:
        # 기존 로그 읽기
        existing_config_logs = []
        if os.path.exists(config_log_file):
            with open(config_log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    try:
                        # JSON 배열 형식으로 파싱 시도
                        if content.startswith('[') and content.endswith(']'):
                            existing_config_logs = json.loads(content)
                        else:
                            # 콤마로 구분된 JSON 객체들을 처리
                            json_str = '[' + content + ']'
                            try:
                                existing_config_logs = json.loads(json_str)
                            except json.JSONDecodeError:
                                # 콤마 없이 개별 JSON 객체들을 분리하여 처리
                                for line in content.split('\n\n'):
                                    if line.strip():
                                        try:
                                            obj = json.loads(line.strip())
                                            existing_config_logs.append(obj)
                                        except json.JSONDecodeError:
                                            print(f"JSON 파싱 오류 발생: {line[:100]}...")
                    except json.JSONDecodeError as e:
                        print(f"설정 로그 파일 파싱 오류: {str(e)}")
        
        # 새 로그를 맨 앞에 추가
        existing_config_logs.insert(0, config_log_data)
        
        # 파일에 저장 - JSON 배열 형식으로 저장
        with open(config_log_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            for i, log_entry in enumerate(existing_config_logs):
                # 각 로그 항목을 들여쓰기된 JSON으로 변환
                formatted_json = json.dumps(log_entry, ensure_ascii=False, indent=2)
                f.write(formatted_json)
                
                # 마지막 항목이 아니면 콤마 추가
                if i < len(existing_config_logs) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write(']\n')
        
        print(f"설정 로그가 {config_log_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"설정 로그 저장 중 오류 발생: {str(e)}")
    
    # 결과 로그 파일에 로그 추가
    try:
        # 기존 로그 읽기
        existing_result_logs = []
        if os.path.exists(result_log_file):
            with open(result_log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    try:
                        # JSON 배열 형식으로 파싱 시도
                        if content.startswith('[') and content.endswith(']'):
                            existing_result_logs = json.loads(content)
                        else:
                            # 콤마로 구분된 JSON 객체들을 처리
                            json_str = '[' + content + ']'
                            try:
                                existing_result_logs = json.loads(json_str)
                            except json.JSONDecodeError:
                                # 콤마 없이 개별 JSON 객체들을 분리하여 처리
                                for line in content.split('\n\n'):
                                    if line.strip():
                                        try:
                                            obj = json.loads(line.strip())
                                            existing_result_logs.append(obj)
                                        except json.JSONDecodeError:
                                            print(f"JSON 파싱 오류 발생: {line[:100]}...")
                    except json.JSONDecodeError as e:
                        print(f"결과 로그 파일 파싱 오류: {str(e)}")
        
        # 새 로그를 맨 앞에 추가
        existing_result_logs.insert(0, result_log_data)
        
        # 파일에 저장 - JSON 배열 형식으로 저장
        with open(result_log_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            for i, log_entry in enumerate(existing_result_logs):
                # 각 로그 항목을 들여쓰기된 JSON으로 변환
                formatted_json = json.dumps(log_entry, ensure_ascii=False, indent=2)
                f.write(formatted_json)
                
                # 마지막 항목이 아니면 콤마 추가
                if i < len(existing_result_logs) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write(']\n')
        
        print(f"결과 로그가 {result_log_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"결과 로그 저장 중 오류 발생: {str(e)}")