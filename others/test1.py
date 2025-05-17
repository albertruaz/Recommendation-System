import numpy as np, scipy.sparse as sp, tempfile, os, scipy.io
from buffalo.data.mm import MatrixMarket, MatrixMarketOptions
from buffalo.algo.options import ALSOption
from buffalo.algo.als import ALS

# 2×3 toy (두 사용자, 한 명은 weight 3/10/25, 다른 한 명은 0)
rows = [0,0,0]
cols = [0,1,2]
vals = [3,10,25]
R = sp.coo_matrix((vals,(rows,cols)), shape=(2,3))

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mtx").name
scipy.io.mmwrite(tmp, R)

mm_opt = MatrixMarketOptions().get_default_option()
mm_opt["input"]["main"] = tmp
mm_opt["data"]["disk_based"] = True   # 에러 회피
data = MatrixMarket(mm_opt); data.create()

als_opt = ALSOption().get_default_option()
als_opt.d = 5; als_opt.alpha = 40.0; als_opt.num_iters = 3; als_opt.num_workers = 1
model = ALS(als_opt, data=data)
model.initialize(); model.train()

print(model.P.dot(model.Q.T)[0])  # 사용자 0 의 예측값
os.remove(tmp)
