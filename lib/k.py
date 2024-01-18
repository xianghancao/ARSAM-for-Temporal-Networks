import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")

    
class K():
    """
    求解K,distance_matrix,adj_matrix
    """
    def __init__(self, log_ret_df):
        """
        log_ret_df：对数收益率面板,行为日期，列为标的，每一列为一条时间序列。
        """
        self.log_ret_df = log_ret_df
        self.date = self.log_ret_df.index[-1]

        
    
    # ------------------------------------------------------------
    def cal_distance_matrix(self):
        """
        计算距离矩阵，根据收益率面板数据df，基于相关系数
        distance_df: 距离矩阵
        """
        corr = self.log_ret_df.corr()
        self.distance_matrix_df = (2*(1-corr)).apply(np.sqrt)

    
    
    # ------------------------------------------------------------
    def cal_adjacent_matrix(self):
        """
        计算邻接矩阵
        adjacent_matrix_df：邻接矩阵
        """
        self.adjacent_matrix_df=(-self.distance_matrix_df).apply(np.exp)
        self.adjacent_matrix_df = self.adjacent_matrix_df - np.identity(self.adjacent_matrix_df.shape[1])
        self.adjacent_matrix_df = self.adjacent_matrix_df.fillna(0)


    
    # ------------------------------------------------------------
    def cal_k(self):
        """
        度中心性
        计算各个节点的度，根据邻接矩阵
        k_df 1维
        """
        k = self.adjacent_matrix_df.sum()
        try:
            tmp = self.k_df.T
            tmp[self.date] = k
            self.k_df = tmp.T
        except:
            self.k_df = pd.DataFrame()
            self.k_df[self.date] = k
            self.k_df = self.k_df.T
        
    