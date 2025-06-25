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
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


class RecommendationRunner:
    """통합 추천 시스템 실행 클래스"""
    
    def __init__(self, config_file='als_config.json'):
        self.logger = setup_logger('recommendation_runner')
        self.config_file = config_file
        
        self.config = _load_config(self.config_file)
        self.db_service = DatabaseService(self.config)
        self.recommendation_service = RecommendationService(self.config)
    
    def run(self):
        try:
            self._log_config_info()
            interactions_df = self.db_service.load_interactions()
            ratings_df = self.db_service.transform_to_ratings(interactions_df)

            recommendations = self.recommendation_service.run_recommendation(ratings_df)
            results = self.db_service.convert_recommendations(recommendations)
            # self.db_service.save_recommendations(results)


            import pandas as pd
            recs_df = pd.DataFrame([
                {'user_id': user_id, 'recommended_items': ','.join(map(str, items))}
                for user_id, items in results.items()
            ])
            recs_df.to_csv('results.csv', index=False, encoding='utf-8')
            self.logger.info("추천 결과를 results.csv 파일로 저장했습니다.")
            
            return results
            
        except Exception as e:
            self.logger.error(f"{str(e)}")
            raise
        finally:
            self._cleanup()
    
    def _log_config_info(self):
        """설정 정보 로깅"""
        self.logger.info("설정 정보")
        self.logger.info(f"데이터 기간: {self.config['days']}일")
        self.logger.info(f"추천 개수: {self.config['top_n']}개")
        self.logger.info(f"모델 반복 횟수: {self.config['max_iter']}")
        self.logger.info(f"모델 잠재 요인 수: {self.config['rank']}")
        self.logger.info(f"테스트 분할: {self.config['split_test_data']}")
        self.logger.info(f"DB 저장: {self.config['save_to_db']} ({self.config['db_type']})")
    
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
        runner = RecommendationRunner(config_file='als_config.json')
        runner.run()
        return 0
        
    except Exception as e:
        logger = setup_logger('main')
        logger.error(f"메인 실행 중 오류: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())