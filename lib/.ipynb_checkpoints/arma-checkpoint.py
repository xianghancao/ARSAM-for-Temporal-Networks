import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA, ARIMA
from tqdm.notebook import tqdm
import datetime
import warnings
warnings.filterwarnings("ignore")

def aic_func(ts, max_ar, max_ma):
    """
    ts为做过差分的时间序列
    """
    aic_dict={}
    for p in range(max_ar+1):
        aic_dict[p] = {}
        for q in range(max_ma+1):
            try:
                model = ARIMA(ts, order=(p, 0, q)).fit()
                aic_dict[p][q] = model.aic
            except:
                aic_dict[p][q] = np.nan
    aic_df = pd.DataFrame(aic_dict).T
    p, q = aic_df.stack().idxmin()
    return p, q




class ARMA():
    def __init__(self, k_df, date, max_ar, max_ma):
        self.k_df = k_df
        self.target = k_df.columns.values
        self.date = date
        self.max_ar = max_ar
        self.max_ma = max_ma
        
    def _processing(self, ts):
        """
        处理时间序列，完成：
        1.nan  2.差分
        """
        ts2 = ts.replace([np.inf, -np.inf, 0], np.nan)
        ts3 = ts2.diff(1)
        ts4 = ts3.fillna(method='ffill').dropna()
        return ts4
    
    
    def cal_p_q(self):
        """
        求多个标的的时序面板的p,q
        """
        p_dict = {}
        q_dict = {}
        for j in self.target:
            ts = self._processing(self.k_df[j])
            p, q = self._cal_p_q(ts)
            p_dict.update({j:p})
            q_dict.update({j:q})
        return p_dict, q_dict
    
            
    # ------------------------------------------------------------  
    def _cal_p_q(self, ts):
        """
        利用AIC准则定阶p
        """
        res = sm.tsa.arma_order_select_ic(ts, ic="aic", max_ar=self.max_ar, max_ma=self.max_ma)
        p, q = res.aic_min_order
        return int(p), int(q)

        
        
    # ------------------------------------------------------------   
    def cal_phi(self, p_dict, q_dict):
        phi_dict={}
        for j in self.target:
            ts = self._processing(self.k_df[j])
            p = p_dict[j]
            q = q_dict[j]
            params = self._cal_phi(ts, p, q)
            phi_dict.update({j:params})
        phi_df = pd.DataFrame(phi_dict)
        phi_df = phi_df.fillna(0)
        return phi_df

            
    # ------------------------------------------------------------    
    def _cal_phi(self, ts, p, q):
        """
        计算系数
        """
        try:
            model = ARIMA(ts, order=(p, 0, q)).fit()
        except:
            p, q = aic_func(ts, max_ar=self.max_ar, max_ma=self.max_ma)
            model = ARIMA(ts, order=(p, 0, q)).fit()
            ts.to_csv('error%s.csv' %str(str(datetime.datetime.now())))
        params = model.params[[i for i in model.params.index if 'ar' in i]]
        params.index = [i.split('.')[1] for i in params.index]
        params =  params.to_dict()
        if 'L1' not in params: params['L1'] = 0
        if 'L2' not in params: params['L2'] = 0
        if 'L3' not in params: params['L3'] = 0
        if 'L4' not in params: params['L4'] = 0
        return params

