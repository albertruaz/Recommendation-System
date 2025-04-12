import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
from buffalo.algo.als import ALS, inited_CUALS
from buffalo.algo.options import ALSOption
import buffalo.data
from buffalo.misc import aux
from buffalo.data.mm import MatrixMarketOptions

def get_df_matrix_mappings(df, row_name, col_name):
    """데이터프레임의 행과 열에 대한 매핑 딕셔너리 생성"""
    rid_to_idx = {}
    idx_to_rid = {}
    
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def df_to_matrix(df, row_name, col_name):
    """데이터프레임을 희소 행렬로 변환"""
    rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid = get_df_matrix_mappings(df, row_name, col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).to_numpy()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).to_numpy()
    V = np.ones(I.shape[0])
    
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def main():
    # 데이터 로드
    data = pd.read_csv('./data/ratings.csv')
    data = data[['userId', 'movieId', 'rating']].astype(str)
    
    # GPU 사용 가능 여부 확인
    print(f"GPU 학습 가능 여부: {inited_CUALS}")
    
    # 행렬 생성 및 저장
    user_items, uid_to_idx, idx_to_uid, mid_to_idx, idx_to_mid = df_to_matrix(data, 'userId', 'movieId')
    mmwrite('./train/main.mtx', user_items)
    
    # ID 리스트 저장
    uid = list(idx_to_uid.values())
    iid = list(idx_to_mid.values())
    
    with open("./train/uid", "w") as f:
        for val in uid:
            print(val, file=f)
    
    with open("./train/iid", "w") as f:
        for val in iid:
            print(val, file=f)
    
    # ALS 옵션 설정
    opt = ALSOption().get_default_option()
    opt.num_workers = 6
    opt.num_iters = 20
    opt.evaluation_period = 20
    opt.evaluation_on_learning = True
    opt.save_best = True
    opt.accelerator = True
    
    # 데이터 옵션 설정
    data_opt = MatrixMarketOptions().get_default_option()
    data_opt.input.main = './train/main.mtx'
    data_opt.input.iid = './train/iid'
    data_opt.input.uid = './train/uid'
    data_opt.data.ath = './train/mm.h5py'
    data_opt.data.validation.p = 0.1
    data_opt.data.validation.max_samples = 5000
    
    # 최적화 설정
    opt.validation = aux.Option({'topk': 10})
    opt.tensorboard = aux.Option({'root': './train/als-validation', 'name': 'als-new'})
    opt.optimize = aux.Option({
        'loss': 'val_ndcg',
        'max_trials': 100,
        'deployment': True,
        'start_with_default_parameters': False,
        'space': {
            'd': ['randint', ['d', 10, 128]],
            'reg_u': ['uniform', ['reg_u', 0.1, 1.0]],
            'reg_i': ['uniform', ['reg_i', 0.1, 1.0]],
            'alpha': ['randint', ['alpha', 1, 10]]
        }
    })
    
    # 모델 초기화 및 최적화
    als = ALS(opt, data_opt=data_opt)
    als.initialize()
    als.opt.model_path = './train/als-best-model.bin'
    als.optimize()
    als.get_optimization_data()
    
    # 학습 데이터 설정
    data_opt = MatrixMarketOptions().get_default_option()
    data_opt.input.main = './train/main.mtx'
    data_opt.input.iid = './train/iid'
    data_opt.input.uid = './train/uid'
    data_opt.data.validation.p = 0.1
    data_opt.data.validation.max_samples = 10000
    data_opt.data.path = './train/mm.h5py'
    
    data = buffalo.data.load(data_opt)
    data.create()
    
    # 최적화된 옵션 로드 및 모델 학습
    del als
    als_opt = ALS()
    als_opt.load('./train/als-best-model.bin')
    
    model = ALS(als_opt.opt, data=data)
    model.initialize()
    model.train()
    
    # 추천 결과 확인
    print("Top 5 movie list for 'userId 1':")
    print(model.topk_recommendation('1', topk=5))
    
    print("\nSimilar movies with 'movieId 4973':")
    print(model.most_similar('4973', topk=5))

if __name__ == "__main__":
    main() 