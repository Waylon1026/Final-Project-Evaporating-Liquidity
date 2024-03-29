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


def generate_table(data, data_m, reproduce=False):
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

    # columns = pd.MultiIndex.from_tuples([
    #     (r'Individual stocks Transaction-price returns', 'Daily', '(1)'),
    #     (r'Individual stocks Transaction-price returns', 'Daily', '(2)'),
    #     (r'Individual stocks Transaction-price returns', 'Daily', '(3)'),
    #     (r'Individual stocks Transaction-price returns', 'Monthly', '(4)'),
    #     (r'Individual stocks Quote-midpoint returns', 'Daily', '(5)'),
    #     (r'Individual stocks Quote-midpoint returns', 'Daily', '(6)'),
    #     (r'Individual stocks Quote-midpoint returns', 'Daily', '(7)'),
    #     (r'Individual stocks Quote-midpoint returns', 'Monthly', '(8)'),
    #     (r'Industry portfolios', 'Daily', '(9)'),
    #     (r'Industry portfolios', 'Daily', '(10)'),
    #     (r'Industry portfolios', 'Daily', '(11)'),
    #     (r'Industry portfolios', 'Monthly', '(12)'),
    # ])

    columns = pd.MultiIndex.from_tuples([
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
    ])

    table.columns = columns
    table_latex = table.to_latex(column_format='l' + 'c' * len(table.columns), 
                                 multicolumn_format='c',
                                 escape=False)
    filename = 'Table_2_reproduce' if reproduce else 'Table_2'
    with open(OUTPUT_DIR / (filename + '.tex'), 'w') as f:
        f.write(table_latex)

    table.to_parquet(DATA_DIR / 'derived' / (filename + '.parquet'))

    return table



if __name__ == "__main__":
    # 2010
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

    # 2023
    reversal_ret = calc_reversal_strategy.load_reversal_return(reproduce=True)
    reversal_ret.columns = ['trade', 'quote', 'industry']
    
    data = prepare_data(reversal_ret, vix, rm)
    data_m = prepare_data(reversal_ret, vix, rm, to_monthly=True)

    table = generate_table(data, data_m, reproduce=True)