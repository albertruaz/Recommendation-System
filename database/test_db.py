import pandas as pd
import json

def load_config():
    with open('config/config.json', 'r') as f:
        return json.load(f)

def convert_ratings():
    # 설정 파일 로드
    config = load_config()
    rating_mapping = config['rating_mapping']
    
    # Input 데이터 로드
    input_df = pd.read_csv('database/input.csv')
    
    # rating 값을 매핑
    input_df['rating'] = input_df['rating'].map(lambda x: rating_mapping[str(int(x))])
    
    # 결과를 interactions 형식으로 저장
    input_df.to_csv('database/interactions.csv', index=False)

if __name__ == "__main__":
    convert_ratings() 