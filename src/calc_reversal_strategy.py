import pandas as pd
import pandas_datareader
import statsmodels.api as sm
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
START_DATE = config.START_DATE
END_DATE = config.END_DATE

import load_FF_industry
import clean_CRSP_stock

def calc_reverse_strategy_ret(df, type_col='industry', ret_col='ret'):
    df['ret-avg'] = df.groupby('date')[ret_col].transform(lambda x: x - x.mean())
    df['w'] = df.groupby('date')['ret-avg'].transform(lambda x: - x / (0.5 * x.abs().sum()))

    for i in range(1, 6):
        df[f'w_lag_{i}'] = df.groupby(type_col)['w'].shift(i) 

    df['rev_ret'] = (df['w_lag_1'] + df['w_lag_2'] + df['w_lag_3'] + df['w_lag_4'] + df['w_lag_5']) * df[ret_col] / 5

    return df.groupby('date')['rev_ret'].sum()


def calc_reverse_strategy_industry(df, start=START_DATE, end=END_DATE):
    df = df.unstack().reset_index()
    df.columns = ["industry", "date", "ret"]

    rev_ret = calc_reverse_strategy_ret(df, 'industry', 'ret')
    return rev_ret[(rev_ret.index>=start) & (rev_ret.index<=end)]


def calc_reverse_strategy_individual(df, ret_col='retx', start=START_DATE, end=END_DATE):
    df[ret_col] *= 100

    rev_ret = calc_reverse_strategy_ret(df, 'permno', ret_col)
    return rev_ret[(rev_ret.index>=start) & (rev_ret.index<=end)]


def calc_beta(factor, fund_ret, constant = True):

    if constant:
        X = sm.tools.add_constant(factor)
    else:
        X = factor

    y = fund_ret
    model = sm.OLS(y, X, missing='drop').fit()
    
    if constant:
        beta = model.params[1]
        
    else:
        beta = model.params

    return beta


def summary_stats(df):
    stats = df.mean().to_frame('Mean return(% per day)')
    stats['Std.dev.(% per day)'] = df.std()
    stats['Skewness'] = df.skew()
    stats['Kurtosis'] = df.kurtosis() + 3
    stats['Worst day return(%)'] = df.min()
    stats['Worst 3-month return(%)'] = df.rolling(63).sum().min()

    index = pd.read_parquet(DATA_DIR / "pulled" / "CRSP_DSIX.parquet")
    index = index.set_index('caldt')['vwretx']*100
    stats['Beta'] = df.apply(lambda x: calc_beta(index, x), axis=0)
    stats['Annualized Sharpe Ratio'] = stats['Mean return(% per day)'] / stats['Std.dev.(% per day)'] * (252**0.5)

    return stats.T


def load_reversal_return(data_dir=DATA_DIR):
    path = Path(data_dir) / "derived" / "reversal_return.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    ff = load_FF_industry.load_FF_industry_portfolio_daily(data_dir=DATA_DIR)[0]
    dfcp = clean_CRSP_stock.load_CRSP_closing_price(data_dir=DATA_DIR)
    dfmid = clean_CRSP_stock.load_CRSP_midpoint(data_dir=DATA_DIR)


    rev_industry = calc_reverse_strategy_industry(ff)
    rev_transact = calc_reverse_strategy_individual(dfcp)
    rev_midpoint = calc_reverse_strategy_individual(dfmid, 'quote_midpoint_return')

    df = pd.concat([rev_transact, rev_midpoint, rev_industry], axis=1)
    df.columns = ['Transact. prices', 'Quote-midpoints', 'Industry portfolio']
    df.to_parquet(DATA_DIR / "derived" / "reversal_return.parquet")

    df_stat = summary_stats(df)
    df_stat.to_parquet(DATA_DIR / "derived" / "Table_1A.parquet")