'''
@Author: Zhiyuan Liu
@Date: 2024-03-08

This file is used to test whether the result from regression_hac.py matches the original Table 2.

Following tests are included:
    - test table formatting
    - test whether the coefficients have the expected sign
    - test whether the coefficients and adjusted R2 match the expected results
'''

import pandas as pd
import numpy as np
import config

DATA_DIR = config.DATA_DIR


def test_table_formatting():
    """
    Test whether the table is formatted correctly
    """
    data = pd.read_parquet(DATA_DIR / 'derived' / 'Table_2.parquet')

    expected_index = ['Intercept', '', 'VIX', '', 'Pre-decim.', '', '$R_M$', '', 'Adj. $R^2$']
    expected_columns = [
            ('\\makecell{Individual stocks\\\\Transaction-price returns}', 'Daily', '(1)'),
        ('\\makecell{Individual stocks\\\\Transaction-price returns}', 'Daily', '(2)'),
        ('\\makecell{Individual stocks\\\\Transaction-price returns}', 'Daily', '(3)'),
        ('\\makecell{Individual stocks\\\\Transaction-price returns}', 'Monthly', '(4)'),
        ('\\makecell{Individual stocks\\\\Quote-midpoint returns}', 'Daily', '(5)'),
        ('\\makecell{Individual stocks\\\\Quote-midpoint returns}', 'Daily', '(6)'),
        ('\\makecell{Individual stocks\\\\Quote-midpoint returns}', 'Daily', '(7)'),
        ('\\makecell{Individual stocks\\\\Quote-midpoint returns}', 'Monthly', '(8)'),
        ('\\makecell{Industry\\\\portfolios}', 'Daily', '(9)'),
        ('\\makecell{Industry\\\\portfolios}', 'Daily', '(10)'),
        ('\\makecell{Industry\\\\portfolios}', 'Daily', '(11)'),
        ('\\makecell{Industry\\\\portfolios}', 'Monthly', '(12)'),
    ]

    assert data.shape == (9, 12)
    assert data.index.tolist() == expected_index
    assert data.columns.tolist() == expected_columns


def test_coef_sign():
    """
    Test whether the coefficients have the expected sign within 1 standard error

    For intercept, no test is performed.
    Ambiguous signs are marked with 1 or -1. Otherwise, the expected sign is marked with 2.
    """
    data = pd.read_parquet(DATA_DIR / 'derived' / 'Table_2.parquet')
    expected_vix_sign = np.ones(12) * 2
    expected_g_sign = np.array([2, 2, 2, 2, 2, 2, 1, 1, 1])
    expected_rm_sign = np.array([-2, -1, -2, -1, -2, -1])

    vix_coefs = data.loc['VIX'].values
    vix_coefs = vix_coefs[vix_coefs != ''].astype('float')
    vix_se = data.iloc[3].str.replace('(', '').str.replace(')', '').values
    vix_se = vix_se[vix_se != ''].astype('float')
    vix_sign = np.sign(vix_coefs + vix_se) + np.sign(vix_coefs - vix_se)

    g_coefs = data.loc['Pre-decim.'].values
    g_coefs = g_coefs[g_coefs != ''].astype('float')
    g_se = data.iloc[5].str.replace('(', '').str.replace(')', '').values
    g_se = g_se[g_se != ''].astype('float')
    g_sign = np.sign(g_coefs + g_se) + np.sign(g_coefs - g_se)

    rm_coefs = data.loc['$R_M$'].values
    rm_coefs = rm_coefs[rm_coefs != ''].astype('float')
    rm_se = data.iloc[7].str.replace('(', '').str.replace(')', '').values
    rm_se = rm_se[rm_se != ''].astype('float')
    rm_sign = np.sign(rm_coefs + rm_se) + np.sign(rm_coefs - rm_se)

    test1 = np.abs(expected_vix_sign - vix_sign) <= 1
    test2 = np.abs(expected_g_sign - g_sign) <= 1
    test3 = np.abs(expected_rm_sign - rm_sign) <= 1

    assert np.all(test1), f'{np.sum(~test1)} VIX coefficients do not have the expected sign within 1 standard error'
    assert np.all(test2), f'{np.sum(~test2)} Pre-decim. coefficients do not have the expected sign within 1 standard error'
    assert np.all(test3), f'{np.sum(~test3)} $R_M$ coefficients do not have the expected sign within 1 standard error'




def test_coef_r2():
    """
    Test whether the results match the expected results within a certain tolerance

    For intercept and standard errors, no test is performed.
    For coefficients of VIX, Pre-decim., and $R_M$, the test is performed using 3 standard errors as the tolerance.
    For Adj. $R^2$, the test is one-sided performed using a tolerance of 0.03.
    """

    # expected_intercept = np.array([-0.03, -0.05, -0.02, 0.02, 
    #                                -0.06, -0.07, -0.04, -0.01, 
    #                                -0.08, -0.09, -0.06, -0.05])
    # intercept_se = np.array([0.03, 0.02, 0.02, 0.02,
    #                          0.03, 0.03, 0.03, 0.02,
    #                          0.02, 0.02, 0.02, 0.01])

    data = pd.read_parquet(DATA_DIR / 'derived' / 'Table_2.parquet')
    multiplier = 3
    
    expected_vix_coefs = np.array([0.22, 0.20, 0.18, 0.15, 
                                   0.16, 0.16, 0.13, 0.10, 
                                   0.07, 0.07, 0.05, 0.04])
    vix_se = np.array([0.02, 0.02, 0.02, 0.01, 
                       0.02, 0.02, 0.02, 0.02, 
                       0.02, 0.02, 0.02, 0.01])
    
    expected_g_coefs = np.array([0.22, 0.22, 0.23, 
                                 0.08, 0.09, 0.09,
                                 0.00, 0.01, 0.01])
    g_se = np.array([0.03, 0.03, 0.03, 
                     0.03, 0.03, 0.03, 
                     0.02, 0.02, 0.02])

    expected_rm_coefs = np.array([-0.60, -0.03, -0.59, -0.16, -0.42, -0.05])
    rm_se = np.array([0.19, 0.26, 0.21, 0.28, 0.17, 0.16])

    expected_adj_r2 = np.array([0.07, 0.11, 0.11, 0.56, 
                                0.03, 0.03, 0.04, 0.25, 
                                0.01, 0.01, 0.01, 0.07])
    adj_r2_tolerance = 0.01

    vix_coefs = data.loc['VIX'].values
    vix_coefs = vix_coefs[vix_coefs != ''].astype('float')

    g_coefs = data.loc['Pre-decim.'].values
    g_coefs = g_coefs[g_coefs != ''].astype('float')

    rm_coefs = data.loc['$R_M$'].values
    rm_coefs = rm_coefs[rm_coefs != ''].astype('float')

    adj_r2 = data.loc['Adj. $R^2$'].values
    adj_r2 = adj_r2[adj_r2 != ''].astype('float')
    
    test1 = np.abs(expected_vix_coefs - vix_coefs) <= (vix_se * multiplier)
    test2 = np.abs(expected_g_coefs - g_coefs) <= (g_se * multiplier)
    test3 = np.abs(expected_rm_coefs - rm_coefs) <= (rm_se * multiplier)
    test4 = (expected_adj_r2 - adj_r2) <= (adj_r2_tolerance * multiplier)

    assert np.all(test1), f'{np.sum(~test1)} VIX coefficients do not match the expected results within the tolerance of {multiplier} standard errors'
    assert np.all(test2), f'{np.sum(~test2)} Pre-decim. coefficients do not match the expected results within the tolerance of {multiplier} standard errors'
    assert np.all(test3), f'{np.sum(~test3)} $R_M$ coefficients do not match the expected results within the tolerance of {multiplier} standard errors'
    assert np.all(test4), f'{np.sum(~test4)} Adj. $R^2$ do not match the expected results within {multiplier*adj_r2_tolerance} tolerance'



# if __name__ == '__main__':
#     table = pd.read_parquet(DATA_DIR / 'derived' / 'Table_2.parquet')

#     test_table_formatting(table)
#     test_coef_sign(table)
#     test_coef_r2(table)