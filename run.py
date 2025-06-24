"""
통합 추천 시스템 실행 파일

모든 설정은 config 파일에서 관리되며, 단순하고 깔끔한 실행을 제공합니다.
"""

import json
import sys
from services.recommendation_service import RecommendationService
from services.db_service import DatabaseService
from utils.logger import setup_logger


def _load_config(config_file: str) -> dict:
    """설정 파일 로드"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 필요한 설정들을 평면화해서 반환
        return {
            # 모델 파라미터
            'model_params': {
                'max_iter': config['pyspark_als']['max_iter'],
                'reg_param': config['pyspark_als']['reg_param'],
                'rank': config['pyspark_als']['rank'],
                'nonnegative': config['pyspark_als'].get('nonnegative', True),
                'cold_start_strategy': config['pyspark_als'].get('cold_start_strategy', 'nan')
            },
            'interaction_weights': config['pyspark_als']['interaction_weights'],
            'interaction_thresholds': config['pyspark_als']['interaction_thresholds'],
            
            # 기본 파라미터
            'days': config['default_params']['days'],
            'top_n': config['default_params']['top_n'],
            
            # 테스트 설정
            'split_test_data': config['testing']['split_test_data'],
            'test_ratio': config['testing']['test_ratio'],
            'random_seed': config['testing']['random_seed'],
            
            # DB 설정
            'save_to_db': config['database']['save_to_db'],
            'db_type': config['database']['db_type']
        }
        
    except FileNotFoundError:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"설정 파일 형식이 잘못되었습니다: {str(e)}")


class RecommendationRunner:
    """통합 추천 시스템 실행 클래스"""
    
    def __init__(self, config_file='als_config.json', log_activate=True):
        self.logger = setup_logger('recommendation_runner')
        self.log_activate = log_activate
        self.config_file = config_file
        
        # 설정 로드
        self.config = _load_config(self.config_file)

        # 서비스 초기화
        self.db_service = DatabaseService(
            db_type=self.config['db_type'],
            interaction_weights=self.config['interaction_weights'],
            interaction_thresholds=self.config['interaction_thresholds']
        )
        self.recommendation_service = RecommendationService(self.config)
    

    
    def run(self):
        """추천 시스템 실행"""
        try:
            if self.log_activate:
                self._log_config_info()
                self.logger.info("데이터 처리중...")
            
            interactions_df = self.db_service.load_interactions(days=self.config['days'])
            self.db_service.prepare_indices(interactions_df)
            ratings_df = self.db_service.transform_to_ratings(interactions_df)
            
            if self.log_activate:
                self.logger.info("추천 생성 시작...")

            # data_loader 설정
            self.recommendation_service.data_loader = self.db_service
            results = self.recommendation_service.run_recommendation(ratings_df)
            
            # self.db_service.save_recommendations(results)

            return results
            
        except Exception as e:
            self.logger.error(f"{str(e)}")
            raise
        finally:
            self._cleanup()
    
    def _log_config_info(self):
        """설정 정보 로깅"""
        self.logger.info("⚙️ 설정 정보:")
        self.logger.info(f"  📅 데이터 기간: {self.config['days']}일")
        self.logger.info(f"  📊 추천 개수: {self.config['top_n']}개")
        self.logger.info(f"  🔧 모델 파라미터: iter={self.config['model_params']['max_iter']}, rank={self.config['model_params']['rank']}")
        self.logger.info(f"  📋 테스트 분할: {self.config['split_test_data']}")
        self.logger.info(f"  💾 DB 저장: {self.config['save_to_db']} ({self.config['db_type']})")
    
    def _cleanup(self):
        """리소스 정리"""
        try:
            self.recommendation_service.model.cleanup()
            self.db_service.cleanup()
        except Exception as e:
            self.logger.warning(f"리소스 정리 중 오류: {str(e)}")


def main():
    """메인 실행 함수"""
    try:
        runner = RecommendationRunner()
        runner.run()
        return 0
        
    except Exception as e:
        logger = setup_logger('main')
        logger.error(f"메인 실행 중 오류: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())