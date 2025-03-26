#!/bin/bash

# 스크립트 설명
echo "============================================================="
echo "  ALS 추천 시스템을 위한 Conda 환경 설정 스크립트"
echo "============================================================="

# Conda 환경 이름 설정
ENV_NAME="recommend-als"

# 기존 환경이 있는지 확인
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "이미 '$ENV_NAME' Conda 환경이 존재합니다."
    read -p "기존 환경을 삭제하고 새로 만드시겠습니까? (y/n): " RECREATE
    if [[ $RECREATE == "y" || $RECREATE == "Y" ]]; then
        echo "기존 환경을 삭제합니다..."
        conda env remove -n $ENV_NAME
    else
        echo "기존 환경을 유지합니다. 필요한 패키지만 설치합니다."
        conda activate $ENV_NAME
        
        # 필수 패키지 설치
        echo "필수 패키지를 설치합니다..."
        conda install -y -c conda-forge implicit pandas numpy scikit-learn joblib matplotlib
        conda install -y -c conda-forge python-dotenv sqlalchemy pymysql psycopg2
        
        read -p "PySpark를 설치하시겠습니까? (y/n): " INSTALL_PYSPARK
        if [[ $INSTALL_PYSPARK == "y" || $INSTALL_PYSPARK == "Y" ]]; then
            echo "PySpark를 설치합니다..."
            conda install -y -c conda-forge pyspark
        fi
        
        exit 0
    fi
fi

# 새 Conda 환경 생성
echo "Python 3.8 기반의 새 Conda 환경 '$ENV_NAME'을 생성합니다..."
conda create -y -n $ENV_NAME python=3.8

# 환경 활성화
echo "환경을 활성화합니다..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 필수 패키지 설치
echo "필수 패키지를 설치합니다..."
conda install -y -c conda-forge implicit pandas numpy scikit-learn joblib matplotlib
conda install -y -c conda-forge python-dotenv sqlalchemy pymysql psycopg2

# PySpark 설치 여부 확인
read -p "PySpark를 설치하시겠습니까? (y/n): " INSTALL_PYSPARK
if [[ $INSTALL_PYSPARK == "y" || $INSTALL_PYSPARK == "Y" ]]; then
    echo "PySpark를 설치합니다..."
    conda install -y -c conda-forge pyspark
fi

# requirements.txt에서 추가 패키지 설치
if [ -f "requirements.txt" ]; then
    echo "requirements.txt에서 추가 패키지를 설치합니다..."
    pip install -r requirements.txt
fi

echo "============================================================="
echo "  설치 완료! 다음 명령어로 환경을 활성화하세요:"
echo "  conda activate $ENV_NAME"
echo "============================================================="
echo ""
echo "추천 시스템 실행 예:"
echo "python main_als.py --mode implicit --user_id 123 --data_source db"
echo "python main_als.py --mode pyspark --csv_path data/ratings.csv"
echo "=============================================================" 