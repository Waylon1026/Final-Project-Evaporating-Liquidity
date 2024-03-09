'''
@Author: Zhiyuan Liu
@Date: 2024-03-08

This file is used to implement the regression using HAC standard errors.
'''

import pandas as pd
import numpy as np
import config
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

import load_vix
import load_CRSP_stock
import calc_reversal_strategy


def prepare_data(ret, vix, rm, lag=5, to_monthly=False):
    """
    Align data for regression

    VIX - divided by sqrt(250), lagged by 5 days
    RM - 4-week return, lagged by 5 days
    g - dummies 1 before 2001/04/09 and 0 after, lagged by 5 days
    """
    ret.dropna(inplace=True)
    vix.dropna(inplace=True)
    rm.dropna(inplace=True)

    ret.index.name = 'date'
    vix.index.name = 'date'
    rm.index.name = 'date'

    vix = vix / np.sqrt(250)

    rm = (1 + rm).rolling(20).apply(np.prod, raw=True) - 1

    g = (rm.index.to_series() <= '2001-04-09').astype(int)
    g.name = 'Pre-decim.'

    if to_monthly:
        ret, vix, rm, g = daily_to_monthly(ret, vix, rm, g, lag)
    else:
        vix = vix.shift(lag)
        rm = rm.shift(lag)
        g = g.shift(lag)

    data = pd.concat([ret, vix, rm, g], axis=1)
    data.dropna(inplace=True)
    return data


def daily_to_monthly(ret, vix, rm, g, lag=5):
    """
    Convert daily data to monthly data

    (Page 24)
    The daily reversal strategy returns are averaged within each calendar month.
    The predictor variables are measured five days before the end of the prior month.
    """
    ret = ret.resample('M').mean()
    # Get the fifth last day of the month and shift to the next month end
    vix = vix.groupby(pd.Grouper(freq='M')).nth(-lag)
    vix.index = vix.index + pd.offsets.MonthEnd()
    vix = vix.shift()
    rm = rm.groupby(pd.Grouper(freq='M')).nth(-lag)
    rm.index = rm.index + pd.offsets.MonthEnd()
    rm = rm.shift()
    g = g.groupby(pd.Grouper(freq='M')).nth(-lag)
    g.index = g.index + pd.offsets.MonthEnd()
    g = g.shift()
    return ret, vix, rm, g
    

def regression_hac(data, y_col, x_cols, maxlags=20):
    """
    Regression with HAC standard errors

    Newey-West HAC standard errors (with 20 lags for daily data and 3 lags for monthly data).
    """
    y = data[y_col]
    X = data[x_cols]
    X = sm.add_constant(X)
    X.rename(columns={'const': 'Intercept'}, inplace=True)
    result = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return result


def regression_all(data, data_m, y_col):
    """
    Run regressions for 4 models
    """
    # VIX
    result_vix = regression_hac(data, y_col, ['VIX'])
    # VIX + g
    result_vix_g = regression_hac(data, y_col, ['VIX', 'Pre-decim.'])
    # VIX + g + RM
    result_vix_g_rm = regression_hac(data, y_col, ['VIX', 'Pre-decim.', '$R_M$'])
    # Monthly
    result_monthly = regression_hac(data_m, y_col, ['VIX', 'Pre-decim.', '$R_M$'], maxlags=3)
    return [result_vix, result_vix_g, result_vix_g_rm, result_monthly]


def generate_table(data, data_m):
    """
    Generate table 2
    """
    results_trade = regression_all(data, data_m, 'trade')
    results_quote = regression_all(data, data_m, 'quote')
    results_industry = regression_all(data, data_m, 'industry')

    table = summary_col(results_trade + results_quote + results_industry,
                        float_format='%0.2f',
                        info_dict={'Adj. $R^2$': lambda x: f'{x.rsquared_adj:0.2f}'},
                        regressor_order=['Intercept', 'VIX', 'Pre-decim.', '$R_M$']
                        ).tables[0]

    table = table.drop(['R-squared', 'R-squared Adj.'], axis=0)

    columns = pd.MultiIndex.from_tuples([
        ('Individual stocks\nTransaction-price returns', 'Daily', '(1)'),
        ('Individual stocks\nTransaction-price returns', 'Daily', '(2)'),
        ('Individual stocks\nTransaction-price returns', 'Daily', '(3)'),
        ('Individual stocks\nTransaction-price returns', 'Monthly', '(4)'),
        ('Individual stocks\nQuote-midpoint returns', 'Daily', '(5)'),
        ('Individual stocks\nQuote-midpoint returns', 'Daily', '(6)'),
        ('Individual stocks\nQuote-midpoint returns', 'Daily', '(7)'),
        ('Individual stocks\nQuote-midpoint returns', 'Monthly', '(8)'),
        ('Industry\nportfolios', 'Daily', '(9)'),
        ('Industry\nportfolios', 'Daily', '(10)'),
        ('Industry\nportfolios', 'Daily', '(11)'),
        ('Industry\nportfolios', 'Monthly', '(12)'),
    ])

    table.columns = columns
    table_latex = table.to_latex()
    with open(OUTPUT_DIR / 'Table_2.tex', 'w') as f:
        f.write(table_latex)

    return table



if __name__ == "__main__":
    reversal_ret = calc_reversal_strategy.load_reversal_return()
    reversal_ret.columns = ['trade', 'quote', 'industry']
    vix = load_vix.load_vix()
    vix.columns = ['VIX']
    rm = load_CRSP_stock.load_CRSP_index_files()
    rm = rm.set_index('caldt')[['vwretx']]
    rm.columns = ['$R_M$']
    
    data = prepare_data(reversal_ret, vix, rm)
    data_m = prepare_data(reversal_ret, vix, rm, to_monthly=True)

    table = generate_table(data, data_m)