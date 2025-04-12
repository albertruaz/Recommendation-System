import pandas as pd
data = pd.read_csv('./data/ratings.csv')
data.head()


# dtype transform
data = data[['userId', 'movieId', 'rating']].astype(str)

# buffalo library import
from buffalo.algo.als import ALS, inited_CUALS
from buffalo.algo.options import ALSOption
import buffalo.data
from buffalo.misc import aux
from buffalo.data.mm import MatrixMarketOptions
import numpy as np
from scipy.io import mmwrite
from scipy.io import mmread
from scipy.sparse import csr_matrix
import scipy.sparse as sp

print(inited_CUALS) # True이면 gpu 학습 가능


# 유저 * 아이템 매트릭스 생성
def get_df_matrix_mappings(df, row_name, col_name):
    
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
    
    rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid = get_df_matrix_mappings(df, row_name, col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).to_numpy()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).to_numpy()
    V = np.ones(I.shape[0])
    
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid
 
# 행렬 매트릭스를 생성하여 파일로 저장
user_items, uid_to_idx, idx_to_uid, mid_to_idx, idx_to_mid = df_to_matrix(data, 'userId', 'movieId')
mmwrite(f'./train/main.mtx', user_items)  

# uid = user ID, iid = item ID
uid = list(idx_to_uid.values())
iid = list(idx_to_mid.values())

# uid, iid 값을 파일로 저장
with open(f"./train/uid", "w") as f:
    for val in uid:
        print(val, file=f)
f.close()

with open(f"./train/iid", "w") as f:
    for val in iid:
        print(val, file=f)
f.close()

# parameter Optimizer
# -- hyperopt 사용하여 최적 파라미터 서치
opt = ALSOption().get_default_option()
opt.num_workers = 6 # worker 수 조정
opt.num_iters = 20
opt.evaluation_period = 20
opt.evaluation_on_learning = True
opt.save_best = True
opt.accelerator = True # GPU option

# optimizer에 사용할 데이터 옵션(경로) 설정
data_opt = MatrixMarketOptions().get_default_option()
data_opt.input.main = './train/main.mtx'
data_opt.input.iid = './train/iid'
data_opt.input.uid = './train/uid'
data_opt.data.ath = './train/mm.h5py'
data_opt.data.validation.p = 0.1
data_opt.data.validation.max_samples = 5000

# optimizer search 범위 설정
opt.validation = aux.Option({'topk' : 10 })
opt.tensorboard = aux.Option({'root' : './train/als-validation', 'name' : 'als-new'})
opt.optimize = aux.Option({
   'loss': 'val_ndcg',
        'max_trials':100,
        'deployment': True,
        'start_with_default_parameters': False,
        'space': {
            'd': ['randint', ['d', 10, 128]],
            'reg_u': ['uniform', ['reg_u', 0.1, 1.0]],
            'reg_i': ['uniform', ['reg_i', 0.1, 1.0]],
            'alpha': ['randint', ['alpha', 1, 10]]
        } 
})

# 설정 옵션을 ALS 모델에 넣고 생성
als = ALS(opt, data_opt = data_opt)
als.initialize()

als.opt.model_path = './train/als-best-model.bin'
als.optimize() # parameter optimizing
als.get_optimization_data()

# 학습시킬 데이터 설정
data_opt = MatrixMarketOptions().get_default_option()
data_opt.input.main = f'./train/main.mtx'
data_opt.input.iid = f'./train/iid'
data_opt.input.uid = f'./train/uid'
data_opt.data.validation.p = 0.1
data_opt.data.validation.max_samples = 10000
data_opt.data.path = f'./train/mm.h5py'

data = buffalo.data.load(data_opt)
data.create()

del als
als_opt = ALS()
als_opt.load('./train/als-best-model.bin') # 최적화 opt 불러오기
als_opt.opt

# model train
model = ALS(als_opt.opt, data= data)
model.initialize()
model.train()

# Top 5 movie list for 'userId 1'
model.topk_recommendation('1',topk=5)

# Simmilar movie with 'movieId 4973'
model.most_similar('4973',topk=5)
