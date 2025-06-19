"""
action_log 데이터 분석 및 코사인 유사도 시각화
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict
from database.mysql_connector import MySQLConnector
from database.vector_db import vector_db
from sqlalchemy import text
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ActionLogAnalyzer:
    """action_log 데이터를 분석하고 코사인 유사도를 계산하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.mysql_connector = MySQLConnector()
        self.vector_db = vector_db()
        
    def get_action_log_data(self) -> pd.DataFrame:
        """
        MySQL에서 product_view 데이터를 가져옵니다.
        전체 데이터에서 LAG 함수로 직전 product_id를 계산한 후
        screen_context = 'detail_similar' 행을 필터링합니다.
        """
        print("product_view 데이터를 가져오는 중...")
        
        with self.mysql_connector.get_session() as session:
            sql = text("""
                SELECT 
                    id,
                    created_at,
                    updated_at,
                    member_id,
                    platform,
                    product_id,
                    screen_context,
                    previous_product_id
                FROM (
                    SELECT 
                        id,
                        created_at,
                        updated_at,
                        member_id,
                        platform,
                        product_id,
                        screen_context,
                        LAG(product_id) OVER (PARTITION BY member_id ORDER BY created_at) AS previous_product_id
                    FROM product_view
                ) AS full_log
                WHERE screen_context = 'detail_similar'
                  AND DAY(created_at) BETWEEN 1 AND 2
                  AND previous_product_id IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 50
            """)
            result = session.execute(sql)
            
            data = []
            for row in result.fetchall():
                data.append({
                    'id': row[0],
                    'created_at': row[1],
                    'updated_at': row[2],
                    'event_name': '',  # product_view에는 없으므로 빈 문자열
                    'member_id': row[3],
                    'platform': row[4],
                    'product_id': row[5],
                    'reference_id': row[7],  # previous_product_id
                    'screen_context': row[6]
                })
        
        df = pd.DataFrame(data)
        print(f"총 {len(df)}개의 product_view 데이터를 가져왔습니다.")
        return df
    
    def get_product_creation_time_from_pg(self, product_id: int) -> datetime:
        """
        PostgreSQL에서 상품의 생성 시간을 가져옵니다.
        """
        session = self.vector_db.Session()
        try:
            sql = text("SELECT created_at FROM product WHERE id = :pid")
            result = session.execute(sql, {'pid': product_id}).fetchone()
            return result[0] if result else None
        finally:
            session.close()
    
    def calculate_cosine_similarity_and_rank(self, action_df: pd.DataFrame) -> pd.DataFrame:
        """
        각 action_log에 대해 product_id와 reference_id 간의 코사인 유사도와 순위를 계산합니다.
        PostgreSQL vector DB에서 직접 계산합니다.
        """
        print("코사인 유사도 및 순위를 계산하는 중...")
        
        results = []
        
        for idx, row in action_df.iterrows():
            product_id = row['product_id']
            reference_id = row['reference_id']
            action_created_at = row['created_at']
            
            if pd.isna(product_id) or pd.isna(reference_id):
                continue
                
            try:
                # PostgreSQL에서 직접 코사인 유사도와 순위 계산
                similarity_result = self.calculate_similarity_from_pg(
                    int(product_id), 
                    int(reference_id), 
                    action_created_at
                )
                
                if similarity_result:
                    cosine_similarity, rank = similarity_result
                    
                    results.append({
                        'action_id': row['id'],
                        'product_id': product_id,
                        'reference_id': reference_id,
                        'created_at': action_created_at,
                        'cosine_similarity': cosine_similarity,
                        'similarity_rank': rank,
                        'event_name': row['event_name'],
                        'member_id': row['member_id']
                    })
                
                if len(results) % 100 == 0:
                    print(f"진행 상황: {len(results)}개 처리 완료")
                    
            except Exception as e:
                print(f"오류 발생 (product_id={product_id}, reference_id={reference_id}): {str(e)}")
                continue
        
        result_df = pd.DataFrame(results)
        print(f"총 {len(result_df)}개의 유사도 데이터를 계산했습니다.")
        return result_df
    
    def calculate_similarity_from_pg(self, product_id: int, reference_id: int, action_created_at: datetime) -> Tuple[float, int]:
        """
        PostgreSQL에서 직접 코사인 유사도와 순위를 계산합니다.
        reference_id를 기준으로 TOP 100 유사 상품을 구하고,
        product_id의 유사도와 TOP 100 내 순위를 반환합니다.
        """
        print(f"\n[DEBUG] 계산 시작 - product_id: {product_id}, reference_id: {reference_id}")
        
        session = self.vector_db.Session()
        try:
            # 먼저 두 상품의 이미지 벡터가 있는지 확인
            check_sql = text("""
                SELECT product_id, vector IS NOT NULL as has_vector
                FROM public.product_image_vector 
                WHERE product_id IN (:pid, :ref_id)
            """)
            check_result = session.execute(check_sql, {
                'pid': product_id,
                'ref_id': reference_id
            }).fetchall()
            
            if not check_result or len(check_result) != 2:
                print(f"[DEBUG] 상품을 찾을 수 없음")
                return None
                
            # 벡터 존재 여부 확인
            for row in check_result:
                if not row[1]:  # has_vector가 False인 경우
                    print(f"[DEBUG] 상품 {row[0]}의 이미지 벡터가 없음")
                    return None
            
            print("[DEBUG] 두 상품 모두 이미지 벡터 있음, 유사도 계산 시작")
            
            # 1. TOP 100 유사 상품 조회 및 target product 존재 여부 확인
            top100_sql = text("""
                WITH ref_vector AS (
                    SELECT vector 
                    FROM public.product_image_vector 
                    WHERE product_id = :ref_id
                )
                SELECT 
                    p.product_id,
                    (p.vector <#> r.vector) AS distance,
                    ROW_NUMBER() OVER (ORDER BY (p.vector <#> r.vector)) AS rank
                FROM public.product_image_vector p
                CROSS JOIN ref_vector r
                WHERE p.product_id != :ref_id
                  AND p.vector IS NOT NULL
                ORDER BY (p.vector <#> r.vector)
                LIMIT 100
            """)
            
            top100_results = session.execute(top100_sql, {
                'ref_id': reference_id
            }).fetchall()
            
            # 2. 두 상품간의 실제 유사도 계산
            similarity_sql = text("""
                WITH ref_vector AS (
                    SELECT vector 
                    FROM public.product_image_vector 
                    WHERE product_id = :ref_id
                )
                SELECT (p.vector <#> r.vector) AS distance
                FROM public.product_image_vector p
                CROSS JOIN ref_vector r
                WHERE p.product_id = :target_id
            """)
            
            similarity_result = session.execute(similarity_sql, {
                'ref_id': reference_id,
                'target_id': product_id
            }).fetchone()
            
            if not similarity_result:
                print("[DEBUG] 유사도를 계산할 수 없음")
                return None
            
            # 실제 유사도 계산
            distance = float(similarity_result[0])
            similarity = 1 - (distance / 2)  # 거리를 유사도로 변환
            
            # TOP 100 결과에서 target product 찾기
            target_rank = None
            for row in top100_results:
                if row[0] == product_id:
                    target_rank = int(row[2])
                    break
            
            # TOP 1, TOP 10, TOP 30, TOP 100 정보 출력
            if len(top100_results) >= 1:
                top1_similarity = 1 - (float(top100_results[0][1]) / 2)
                print(f"[DEBUG] TOP 1 유사도: {top1_similarity:.4f}")
            
            if len(top100_results) >= 10:
                top10_similarity = 1 - (float(top100_results[9][1]) / 2)
                print(f"[DEBUG] TOP 10 유사도: {top10_similarity:.4f}")
            
            if len(top100_results) >= 30:
                top30_similarity = 1 - (float(top100_results[29][1]) / 2)
                print(f"[DEBUG] TOP 30 유사도: {top30_similarity:.4f}")
            
            if len(top100_results) >= 100:
                top100_similarity = 1 - (float(top100_results[99][1]) / 2)
                print(f"[DEBUG] TOP 100 유사도: {top100_similarity:.4f}")
            
            if target_rank:
                # TOP 100에 있는 경우
                print(f"[DEBUG] TOP 100 안에 있음 - 실제 유사도: {similarity:.4f}, TOP 100 순위: {target_rank}")
                return similarity, target_rank
            else:
                # TOP 100에 없는 경우
                print(f"[DEBUG] TOP 100 밖 - 실제 유사도: {similarity:.4f}")
                return similarity, 0  # 0은 TOP 100 밖을 의미
            
        except Exception as e:
            print(f"[DEBUG] 오류 발생: {str(e)}")
            print(f"PostgreSQL 유사도 계산 중 오류: {str(e)}")
            return None
        finally:
            session.close()
    
    def create_similarity_distribution_plot(self, similarity_df: pd.DataFrame, save_path: str = None):
        """
        코사인 유사도 분포 그래프를 생성합니다.
        """
        print("유사도 분포 그래프를 생성하는 중...")
        
        plt.figure(figsize=(12, 8))
        
        # 서브플롯 생성
        plt.subplot(2, 2, 1)
        plt.hist(similarity_df['cosine_similarity'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Cosine Similarity Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_sim = similarity_df['cosine_similarity'].mean()
        median_sim = similarity_df['cosine_similarity'].median()
        plt.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
        plt.axvline(median_sim, color='green', linestyle='--', label=f'Median: {median_sim:.3f}')
        plt.legend()
        
        # 순위 분포 (유효한 순위만)
        plt.subplot(2, 2, 2)
        rank_data = similarity_df[similarity_df['similarity_rank'] > 0]['similarity_rank']
        if len(rank_data) > 0:
            plt.hist(rank_data, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.title('Similarity Rank Distribution')
            plt.xlabel('Similarity Rank')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No valid rank data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Similarity Rank Distribution')
        
        # 유사도 vs 순위 산점도
        plt.subplot(2, 2, 3)
        valid_data = similarity_df[similarity_df['similarity_rank'] > 0]
        if len(valid_data) > 0:
            plt.scatter(valid_data['cosine_similarity'], valid_data['similarity_rank'], 
                       alpha=0.6, color='purple', s=10)
            plt.title('Cosine Similarity vs Rank')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Similarity Rank')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No valid rank data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Cosine Similarity vs Rank')
        
        # 이벤트별 유사도 분포
        plt.subplot(2, 2, 4)
        event_types = similarity_df['event_name'].unique()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, event in enumerate(event_types):
            event_data = similarity_df[similarity_df['event_name'] == event]['cosine_similarity']
            plt.hist(event_data, bins=30, alpha=0.5, 
                    color=colors[i % len(colors)], label=event)
        plt.title('Similarity Distribution by Event Type')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        if save_path is None:
            save_path = f"similarity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프가 저장되었습니다: {save_path}")
        
        # 통계 정보 출력
        print("\n=== 분석 결과 ===")
        print(f"총 데이터 수: {len(similarity_df)}")
        print(f"평균 코사인 유사도: {similarity_df['cosine_similarity'].mean():.4f}")
        print(f"중간값 코사인 유사도: {similarity_df['cosine_similarity'].median():.4f}")
        print(f"표준편차: {similarity_df['cosine_similarity'].std():.4f}")
        
        valid_ranks = similarity_df[similarity_df['similarity_rank'] > 0]['similarity_rank']
        if len(valid_ranks) > 0:
            print(f"평균 순위: {valid_ranks.mean():.1f}")
            print(f"중간값 순위: {valid_ranks.median():.1f}")
            print(f"유효한 순위 데이터: {len(valid_ranks)}개")
        else:
            print("유효한 순위 데이터가 없습니다.")
        
        print(f"\n이벤트별 데이터 수:")
        print(similarity_df['event_name'].value_counts())
        
        return save_path
    
    def run_analysis(self):
        """전체 분석을 실행합니다."""
        print("=== action_log 코사인 유사도 분석 시작 ===")
        
        try:
            # 1. action_log 데이터 가져오기
            action_df = self.get_action_log_data()
            
            if len(action_df) == 0:
                print("분석할 데이터가 없습니다.")
                return
            
            # 2. 코사인 유사도 및 순위 계산
            similarity_df = self.calculate_cosine_similarity_and_rank(action_df)
            
            if len(similarity_df) == 0:
                print("유사도를 계산할 수 있는 데이터가 없습니다.")
                return
            
            # 3. 결과를 CSV로 저장
            csv_path = f"similarity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            similarity_df.to_csv(csv_path, index=False)
            print(f"결과가 CSV로 저장되었습니다: {csv_path}")
            
            # 4. 그래프 생성
            plot_path = self.create_similarity_distribution_plot(similarity_df)
            
            print("=== 분석 완료 ===")
            return similarity_df, plot_path
            
        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            raise
        finally:
            # 연결 해제
            self.mysql_connector.close()
            self.vector_db.close()

def main():
    """메인 실행 함수"""
    analyzer = ActionLogAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 