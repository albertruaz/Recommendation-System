"""
Spark 관련 유틸리티 모듈

이 모듈은 Spark 세션 관리를 제공합니다.
"""

import logging
from pyspark.sql import SparkSession


class SparkSingleton:
    """Spark 세션 싱글톤 관리 클래스"""
    _session = None
    
    @staticmethod
    def get(app_name="RecoApp", log_level="ERROR"):
        """
        Spark 세션을 가져오거나 생성합니다 (싱글톤 패턴)
        
        Args:
            app_name: Spark 애플리케이션 이름
            log_level: 로그 레벨 (ERROR, WARN, INFO)
            
        Returns:
            SparkSession: 생성된 또는 기존의 SparkSession
        """
        # 이미 세션이 있고 활성 상태인 경우 재사용
        if SparkSingleton._session is not None:
            try:
                # 간단한 작업으로 세션 활성 상태 확인
                SparkSingleton._session.sparkContext.parallelize([1]).count()
                return SparkSingleton._session
            except Exception:
                # 오류 발생 시 세션이 비활성화된 것으로 간주
                SparkSingleton._session = None
        
        # 새 세션 생성
        SparkSingleton._session = (
            SparkSession.builder
            .appName(app_name)
            # 메모리 설정
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.memory.fraction", "0.7")
            .config("spark.memory.storageFraction", "0.3")
            # 네트워크 설정
            .config("spark.network.timeout", "1200s")
            .config("spark.driver.maxResultSize", "1g")
            .config("spark.rpc.message.maxSize", "256")
            # 성능 설정
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.default.parallelism", "4")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.adaptive.enabled", "true")
            # UI 설정
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        
        # 로그 레벨 설정
        SparkSingleton._session.sparkContext.setLogLevel(log_level)
        
        return SparkSingleton._session
    
    @staticmethod
    def is_active(session):
        """
        주어진 Spark 세션이 활성 상태인지 확인합니다.
        
        Args:
            session: 확인할 SparkSession 객체
            
        Returns:
            bool: 세션이 활성 상태이면 True, 아니면 False
        """
        if session is None:
            return False
            
        try:
            # 1. isStopped 메서드 먼저 시도 (새 버전)
            if hasattr(session.sparkContext, 'isStopped'):
                return not session.sparkContext.isStopped
                
            # 2. 간단한 RDD 작업 시도 (이전 버전)
            test_rdd = session.sparkContext.parallelize([1, 2, 3])
            count = test_rdd.count()
            return count == 3
        except Exception as e:
            logging.debug(f"세션 활성화 확인 중 오류: {str(e)}")
            return False
    
    @staticmethod
    def stop():
        """현재 Spark 세션을 안전하게 종료합니다."""
        if SparkSingleton._session is not None:
            try:
                # 1. 캐시된 DataFrame들 정리
                SparkSingleton._session.catalog.clearCache()
                # 2. 세션 종료
                SparkSingleton._session.stop()
                SparkSingleton._session = None
                logging.info("SparkSession 종료 완료")
            except Exception as e:
                logging.warning(f"SparkSession 종료 중 오류: {e}")
    
    @staticmethod
    def cleanup():
        """완전한 리소스 정리"""
        SparkSingleton.stop() 