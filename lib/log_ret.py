import pandas as pd
import os
import numpy as np

def load_data(path):
    print('load_data', path)
    df = pd.read_excel(path, index_col='交易时间')
    df.dropna(inplace=True)
    return df

class LogRet():
    def __init__(self, start_date, end_date, store_path=None):
        self.start_date = start_date
        self.end_date = end_date
        self.store_path = store_path
    
    def main(self):
        self._load()
        self._log()
        self._select()
        
    def _log(self):
        # 对数收益率
        self.log_ret_df = (self.cleaned_df/self.cleaned_df.shift(1)).apply(np.log)
        self.log_ret_df = self.log_ret_df.dropna()
        self.log_ret_df.index = [str(i)[:10] for i in self.log_ret_df.index]
        if self.store_path is not None:
            self.log_ret_df.to_csv('%s/log_ret.csv' %self.store_path)
        
    def _select(self):
        # 选取指定区间
        self.log_ret_df = self.log_ret_df.loc[self.start_date:self.end_date]
        
        
    def _load(self):
        path = 'datasets'
        list_df = pd.read_excel(os.path.join(path, '指数数据表.xlsx'), dtype={'指数编码':np.str_}, index_col='指数编码')
        list_df.index = list_df.index.astype(np.str_)
        #list_df = list_df[list_df['select'] == 'v']
        list_dict = list_df['指数名称'].to_dict()
        self.list_dict = list_dict
        print(list_dict)

        # 第一批国家
        origin_df=pd.DataFrame()
        path = 'datasets/指数日线数据/'
        for i in os.listdir(path):
            code = i.split('.xls')[0].split('K线导出_')[-1].split('_日线数据')[0]
            if code not in list_dict: continue
            origin_df[list_dict[code]] = load_data(os.path.join(path, i))['收盘价']

        path = 'datasets/investing/'
        for i in os.listdir(path):
            if 'csv' not in i: continue
            code = i.split('.csv')[0]
            if code not in list_dict: continue
            print('load_data', os.path.join(path, i))
            df = pd.read_csv('%s/%s' %(path, i))
            df.index = pd.to_datetime(df['日期'])
            df['收盘'] = [float(i.replace(',', '')) for i in df['收盘']]
            df = df.sort_index()
            origin_df[list_dict[i.split('.csv')[0]]] = df['收盘']

        # 数据填充
        self.cleaned_df = origin_df.fillna(method="ffill").dropna()
        if self.store_path is not None:
            self.cleaned_df.to_csv('%s/price.csv' %self.store_path)
        
