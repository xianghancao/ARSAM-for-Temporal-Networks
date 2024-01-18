
import numpy as np
import pandas as pd


# ------------------------------------------------------------   
def cal_w_matrix(params):
    """
    w_matrix: 关联矩阵W
    """
    w_matrix_t_1_df = pd.DataFrame(np.diag((params.loc['L1']-1).values),
                               index=params.columns.values,
                               columns=params.columns.values)


    w_matrix_t_2_df = pd.DataFrame(np.diag((params.loc['L2']-params.loc['L1']).values),
                               index=params.columns.values,
                               columns=params.columns.values)


    w_matrix_t_3_df = pd.DataFrame(np.diag((params.loc['L3']-params.loc['L2']).values),
                               index=params.columns.values,
                               columns=params.columns.values)


    w_matrix_t_4_df = pd.DataFrame(np.diag((-1 * params.loc['L4']).values),
                               index=params.columns.values,
                               columns=params.columns.values)
    return {'L1':w_matrix_t_1_df, 'L2':w_matrix_t_2_df, 'L3':w_matrix_t_3_df, 'L4':w_matrix_t_4_df}
