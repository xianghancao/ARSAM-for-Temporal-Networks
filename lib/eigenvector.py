import os, sys
import numpy as np
import pandas as pd

def principle_eigen(matrix):
    eigenvalue, eigenmatrix = np.linalg.eig(matrix)
    principle_eigenvalue = eigenvalue[0]
    principle_eigenvector = eigenmatrix[:,0]
    return principle_eigenvalue, principle_eigenvector


def normalization(arr):
    norm = np.linalg.norm(arr, ord=2)
    return np.abs(arr)/norm


def load_w_matrix(store_path, date):
    w_matrix = {}
    for k in np.sort(os.listdir('%s/%s/w_matrix/%s' %(sys.path[0], store_path,date))):
        if '.ipynb_checkpoints' in k:continue
        w_matrix[k.split('.')[0]] = pd.read_csv('%s/%s/w_matrix/%s/%s' %(sys.path[0], store_path, date, k), index_col=0)
    return w_matrix

            
def load_adjacent_matrix(store_path, date):
    adjacent_matrix = pd.read_csv('%s/%s/adjacent_matrix/%s.csv' %(sys.path[0], store_path, date), index_col=0)
    return adjacent_matrix
                                  

class Eigenvector():
    def __init__(self, sliding_window, store_path):
        print('Eigenvector initialize...')
        self.sliding_window = sliding_window
        self.date = self.sliding_window[-1]
        self.sliding_window_length = len(self.sliding_window)
        self.store_path = store_path
        

    def _load_data(self):
        self.adjacent_matrix_dict = {}
        self.w_matrix_dict = {}
        for date in self.sliding_window:
            self.adjacent_matrix_dict[date] = load_adjacent_matrix(self.store_path, date)
            self.w_matrix_dict[date] = load_w_matrix(self.store_path, date)
            
    def cal_eigenvector_adj_matrix(self):
        print('cal_eigenvector_adj_matrix()...')
        self._load_data()
        eigenval_ts, eigenvector_ts={}, {}
        for date in self.sliding_window:
            eigenvalue, eigenvector = principle_eigen(self.adjacent_matrix_dict[date].values)
            eigenvector_ts[date] = eigenvector
            eigenval_ts[date] = eigenvalue
        self.eigenvector_adj_matrix_df = pd.DataFrame(eigenvector_ts, columns=self.sliding_window).T
        self.eigenval_adj_matrix_df = pd.Series(eigenval_ts)

        
        
    def cal_eigenvector_super_adj_matrix(self):  
        print('cal_eigenvector_super_adj_matrix()...')
        val_df = self.eigenval_adj_matrix_df.copy()
        max_index = val_df.argmax()
        vector_df = self.eigenvector_adj_matrix_df.copy()
        vector_df.iloc[:max_index,:] = 0
        max_date = val_df.index[max_index]
        max_lambda = val_df.loc[max_date]


        # 某一窗口循环求解特征向量
        for j in range(max_index+1, ):
            today = val_df.index[j]
            yesterday = val_df.index[j-1]
            a = self.adjacent_matrix_dict[today].values
            l = np.matrix(np.eye(21) * max_lambda)
            w1 = np.matrix(self.w_matrix_dict[today]['L1']*-1.)
            w2 = np.matrix(self.w_matrix_dict[today]['L2']*-1.)
            w3 = np.matrix(self.w_matrix_dict[today]['L3']*-1.)
            w4 = np.matrix(self.w_matrix_dict[today]['L4']*-1.)
            v1 = np.matrix(vector_df.iloc[j-1]).T
            v2 = np.matrix(vector_df.iloc[j-2]).T
            v3 = np.matrix(vector_df.iloc[j-3]).T
            v4 = np.matrix(vector_df.iloc[j-4]).T
            vector_df.iloc[j]  = np.array((l-a).I * (w1 * v1 + w2*v2 + w3*v3 + w4*v4)).flatten()

        # 归一化
        for j in range(max_index, self.sliding_window_length):
            vector_df.iloc[j] = normalization(vector_df.iloc[j].values)
        vector_df.columns = self.adjacent_matrix_dict[today].columns
        
        self.vector_df = vector_df
        # 计算某一窗口各个国家的平均特征向量值
        self.avg_vector = vector_df.iloc[max_index:].mean(axis=0)

        self.avg_vector.to_csv('%s/eigenvector/%s.csv' %(self.store_path, self.date))
        


