import os, sys
from tqdm import tqdm
import numpy as np
import pandas as pd

from lib.log_ret import LogRet
from lib.k import K
from lib.arma import ARMA
from lib.w import cal_w_matrix
from lib.eigenvector import Eigenvector

import json
def save_json(data, path):
    if '.json' not in path: raise Exception('pls specify a .json file in path!')
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(data, f)
    return 

def load_json(path):
    if '.json' not in path: raise Exception('json file load failed!')
    with open(path, 'r+', encoding='utf-8') as f:
        data = json.load(f)
    return data


def sliding_index(df, sliding_window_len):
    return range(sliding_window_len-1, df.shape[0])


def sliding(df, sliding_window_len, i):
    sliding_df = df.iloc[i+1-sliding_window_len:i+1, :]
    date = str(df.index.values[i])[:10]
    return sliding_df, date


def rolling_k(log_ret_df, sliding_window_len, store_path):
    print('rolling_k...')
    k_df = pd.DataFrame()
    for i in tqdm(sliding_index(log_ret_df, sliding_window_len)):
        sliding_log_ret_df, date = sliding(log_ret_df, sliding_window_len, i)
        k = K(sliding_log_ret_df)
        k.cal_distance_matrix()
        k.cal_adjacent_matrix()
        k.cal_k()
        adjacent_matrix_dict = k.adjacent_matrix_df
        k_df = k_df.append(k.k_df)
        if not os.path.exists('%s/adjacent_matrix' %store_path): os.mkdir('%s/adjacent_matrix' %store_path)
        adjacent_matrix_dict.to_csv('%s/adjacent_matrix/%s.csv' %(store_path, date))
    k_df.to_csv('%s/k.csv' %store_path)
    return k_df



# ------------------------------------------------------------
def rolling_w(k_df, sliding_window_len, max_ar, max_ma, store_path, cache=True):
    # 计算ARMA的phi系数
    print('rolling_w...')
    p_dict, q_dict = dict(), dict()
    for i in tqdm(sliding_index(k_df, sliding_window_len)):
        sliding_k_df, date = sliding(k_df, sliding_window_len, i)

        p_json_path = os.path.join(store_path, 'ARMA_p', date+'.json')
        q_json_path = os.path.join(store_path, 'ARMA_q', date+'.json')                
        if cache and os.path.exists(p_json_path):
            print('find', date, 'in the cache')
            p_dict.update(load_json(p_json_path))
            q_dict.update(load_json(q_json_path))
        elif (cache and not os.path.exists(p_json_path)) or not cache:
            print('not use cache OR find', date, 'not in the cache')
            arma = ARMA(sliding_k_df, date, max_ar=max_ar, max_ma=max_ma)
            p, q = arma.cal_p_q()
            p_dict.update({date:p})
            q_dict.update({date:q})
            print({date:p})
            save_json(data={date:p}, path=p_json_path)
            save_json(data={date:q}, path=q_json_path)

        arma = ARMA(sliding_k_df, date, max_ar=max_ar, max_ma=max_ma)
        param_df = arma.cal_phi(p_dict[date], q_dict[date])
        if not os.path.exists('%s/phi/' %store_path): os.mkdir('%s/phi/' %(store_path))
        param_df.to_csv('%s/phi/%s.csv' %(store_path, date))
        # --------------------
        w_matrix = cal_w_matrix(param_df)
        if not os.path.exists('%s/w_matrix/' %store_path): os.mkdir('%s/w_matrix/' %store_path)
        if not os.path.exists('%s/w_matrix/%s' %(store_path, date)): os.mkdir('%s/w_matrix/%s' %(store_path, date))
        print(date)
        for k in w_matrix:
            w_matrix[k].to_csv('%s/w_matrix/%s/%s.csv' %(store_path, date, k))
    return w_matrix
            
        
    

store_path='output/21nodes_200day_p6_q2'
start_date = '2004-01-01'
end_date = '2022-05-05'
rolling_k_window_len = 200
rolling_p_window_len = 200
rolling_w_window_len = 200
sliding_e_window_len = 200
# ----------------------------------------------------------------
lr = LogRet(start_date, end_date, store_path=store_path)
lr.main()
log_ret_df = lr.log_ret_df


# ----------------------------------------------------------------
k_df = rolling_k(log_ret_df, rolling_k_window_len, store_path)

#----------------------------------------------------------------
w_matrix = rolling_w(k_df, 
          sliding_window_len=rolling_w_window_len,
          max_ar=4, max_ma=2,
          store_path=store_path,
          cache=True)


#----------------------------------------------------------------
for i in tqdm(sliding_index(k_df, sliding_e_window_len)):
    sliding_k_df, date = sliding(k_df, sliding_e_window_len, i)
    sliding_window = sliding_k_df.index
    m = Eigenvector(sliding_window, store_path=store_path)
    m.cal_eigenvector_adj_matrix()
    m.cal_eigenvector_super_adj_matrix()