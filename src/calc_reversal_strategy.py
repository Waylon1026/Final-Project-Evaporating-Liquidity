import pandas as pd
import pandas_datareader
import statsmodels.api as sm
import config
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(config.DATA_DIR)
START_DATE = config.START_DATE
END_DATE = config.END_DATE

import load_FF_industry
import load_CRSP_stock
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


def calc_multiple_beta(factor, fund_ret, constant = True):

    if constant:
        X = sm.tools.add_constant(factor)
    else:
        X = factor

    y = fund_ret
    model = sm.OLS(y, X, missing='drop').fit()
    
    if constant:
        beta = model.params[1:]
        
    else:
        beta = model.params

    return beta


def calc_hedged_return(ret):
    
    index = load_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)
    index = index.set_index('caldt')['vwretx']*100
    shifted_sign = index.shift(1).apply(lambda x: 1 if x > 0 else -1)
    index_shifted_sign = index * shifted_sign

    factor = pd.concat([index, index_shifted_sign], axis=1)

    beta = calc_multiple_beta(factor, ret)
    time_varying_beta = beta[0] + beta[1] * shifted_sign

    hedged_ret = ret - time_varying_beta * index

    return hedged_ret


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

    stats = stats.applymap('{:.2f}'.format)

    return stats.T


def load_reversal_return(data_dir=DATA_DIR, hedged=False):
    if hedged:
        path = Path(data_dir) / "derived" / "reversal_return_hedged.parquet"
    else:
        path = Path(data_dir) / "derived" / "reversal_return.parquet"
    return pd.read_parquet(path)


def load_Table_1A(data_dir=DATA_DIR, reproduce=False):
    """
    Load Table 1A: Summary Statistics of Reversal Strategy Returns
    """
    if reproduce:
        path = Path(data_dir) / "derived" / "Table_1A_reproduce.parquet"
    else:
        path = Path(data_dir) / "derived" / "Table_1A.parquet"

    df = pd.read_parquet(path)
    return df


def load_Table_1B(data_dir=DATA_DIR, reproduce=False):
    """
    Load Table 1B: Summary Statistics of Hedged Reversal Strategy Returns
    """
    if reproduce:
        path = Path(data_dir) / "derived" / "Table_1B_reproduce.parquet"
    else:
        path = Path(data_dir) / "derived" / "Table_1B.parquet"
    df = pd.read_parquet(path)
    return df


def demo():
    df_1A = load_Table_1A(data_dir=DATA_DIR, reproduce=False)
    df_1B = load_Table_1B(data_dir=DATA_DIR, reproduce=False)


if __name__ == "__main__":
    ff = load_FF_industry.load_FF_industry_portfolio_daily(data_dir=DATA_DIR)[0]
    dfcp = clean_CRSP_stock.load_CRSP_closing_price(data_dir=DATA_DIR)
    dfmid = clean_CRSP_stock.load_CRSP_midpoint(data_dir=DATA_DIR)


    # Replicate Table 1A
    rev_industry = calc_reverse_strategy_industry(ff)
    rev_transact = calc_reverse_strategy_individual(dfcp)
    rev_midpoint = calc_reverse_strategy_individual(dfmid, 'quote_midpoint_return')

    ret_raw = pd.concat([rev_transact, rev_midpoint, rev_industry], axis=1)
    ret_raw.columns = ['Transact. prices', 'Quote-midpoints', 'Industry portfolio']
    ret_raw.to_parquet(DATA_DIR / "derived" / "reversal_return.parquet")

    ret_raw = load_reversal_return(data_dir=DATA_DIR)

    df_stat_A = summary_stats(ret_raw)
    df_stat_A = df_stat_A.style.format('{:.2f}',na_rep='')
    df_stat_A.to_parquet(DATA_DIR / "derived" / "Table_1A.parquet")


    # Reproduce Table 1A
    # df_stat_A_new.to_parquet(DATA_DIR / "derived" / "Table_1A_reproduce.parquet")


    # Replicate Table 1B
    hedged_rev_transact = calc_hedged_return(rev_transact)
    hedged_rev_midpoint = calc_hedged_return(rev_midpoint)
    hedged_rev_industry = calc_hedged_return(rev_industry)

    ret_hedged = pd.concat([hedged_rev_transact, hedged_rev_midpoint, hedged_rev_industry], axis=1)
    ret_hedged.columns = ['Hedged Transact. prices', 'Hedged Quote-midpoints', 'Hedged Industry portfolio']
    ret_hedged.to_parquet(DATA_DIR / "derived" / "reversal_return_hedged.parquet")

    ret_hedged = load_reversal_return(data_dir=DATA_DIR, hedged=True)
    
    df_stat_B = summary_stats(ret_hedged)
    df_stat_B = df_stat_B.style.format('{:.2f}',na_rep='')
    df_stat_B.to_parquet(DATA_DIR / "derived" / "Table_1B.parquet")


    # Reproduce Table 1B
    # df_stat_B_new.to_parquet(DATA_DIR / "derived" / "Table_1B_reproduce.parquet")