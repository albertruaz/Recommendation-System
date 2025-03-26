"""
설정 관련 유틸리티 함수
"""

import os
import json
import logging
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    모델 설정 파일을 로드합니다.
    
    Args:
        config_path (str, optional): 설정 파일 경로. 기본값은 configs/model_config.json
        
    Returns:
        Dict[str, Any]: 설정 딕셔너리
    """
    if config_path is None:
        config_path = os.path.join('configs', 'model_config.json')
        
    try:
        if not os.path.exists(config_path):
            # 기본 설정 반환
            return {
                'als': {
                    'factors': 10,
                    'regularization': 0.1,
                    'iterations': 15,
                    'alpha': 1.0,
                    'use_gpu': False
                }
            }
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        return config
        
    except Exception as e:
        logging.warning(f"설정 파일 로드 중 오류 발생: {str(e)}")
        logging.warning("기본 설정을 사용합니다.")
        return {
            'als': {
                'factors': 10,
                'regularization': 0.1,
                'iterations': 15,
                'alpha': 1.0,
                'use_gpu': False
            }
        } 