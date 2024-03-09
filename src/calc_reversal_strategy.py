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


def calc_hedged_return(ret, reproduce=False):
    
    index = load_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)
    if reproduce:
        index = index[index['caldt'] <= '2023-12-31']
    else:
        index = index[index['caldt'] <= '2010-12-31']
    index = index.set_index('caldt')['vwretx']*100
    shifted_sign = index.shift(1).apply(lambda x: 1 if x > 0 else -1)
    index_shifted_sign = index * shifted_sign

    factor = pd.concat([index, index_shifted_sign], axis=1)

    beta = calc_multiple_beta(factor, ret)
    time_varying_beta = beta[0] + beta[1] * shifted_sign

    hedged_ret = ret - time_varying_beta * index

    return hedged_ret


def summary_stats(df, reproduce=False):
    stats = df.mean().to_frame('Mean return(% per day)')
    stats['Std.dev.(% per day)'] = df.std()
    stats['Skewness'] = df.skew()
    stats['Kurtosis'] = df.kurtosis() + 3
    stats['Worst day return(%)'] = df.min()
    stats['Worst 3-month return(%)'] = df.rolling(63).sum().min()

    index = load_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)
    if reproduce:
        index = index[index['caldt'] <= '2023-12-31']
    else:
        index = index[index['caldt'] <= '2010-12-31']
    index = index.set_index('caldt')['vwretx']*100
    stats['Beta'] = df.apply(lambda x: calc_beta(index, x), axis=0)
    stats['Annualized Sharpe Ratio'] = stats['Mean return(% per day)'] / stats['Std.dev.(% per day)'] * (252**0.5)

    stats = stats.applymap('{:.2f}'.format)

    return stats.T


def load_reversal_return(data_dir=DATA_DIR, hedged=False, reproduce=False):
    """
    Load reversal strategy returns

    Args:
        - hedged: whether to load hedged returns
        - reproduce: whether to load the returns until end date of data used in reproduction
    """
    if hedged:
        if reproduce:
            path = Path(data_dir) / "derived" / "reversal_return_hedged_2023.parquet"
        else:
            path = Path(data_dir) / "derived" / "reversal_return_hedged_2010.parquet"
    else:
        if reproduce:
            path = Path(data_dir) / "derived" / "reversal_return_2023.parquet"
        else:
            path = Path(data_dir) / "derived" / "reversal_return_2010.parquet"
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
    ret_raw = load_reversal_return(data_dir=DATA_DIR, hedged=False)
    ret_hedged = load_reversal_return(data_dir=DATA_DIR, hedged=True)
    
    df_1A = load_Table_1A(data_dir=DATA_DIR, reproduce=False)
    df_1B = load_Table_1B(data_dir=DATA_DIR, reproduce=False)
    df_1A_new = load_Table_1A(data_dir=DATA_DIR, reproduce=True)
    df_1B_new = load_Table_1B(data_dir=DATA_DIR, reproduce=True)


if __name__ == "__main__":
    ff = load_FF_industry.load_FF_industry_portfolio_daily(data_dir=DATA_DIR)[0]
    dfcp = clean_CRSP_stock.load_CRSP_closing_price(data_dir=DATA_DIR)
    dfmid = clean_CRSP_stock.load_CRSP_midpoint(data_dir=DATA_DIR)

    ff_2010 = ff[(ff.index >= '1998-01-01') & (ff.index <= '2010-12-31')]
    dfcp_2010 = dfcp[(dfcp['date'] >= '1998-01-01') & (dfcp['date'] <= '2010-12-31')]
    dfmid_2010 = dfmid[(dfmid['date'] >= '1998-01-01') & (dfmid['date'] <= '2010-12-31')]


    # Replicate Table 1A
    ret_industry = calc_reverse_strategy_industry(ff_2010)
    ret_transact = calc_reverse_strategy_individual(dfcp_2010)
    ret_midpoint = calc_reverse_strategy_individual(dfmid_2010, 'quote_midpoint_return')

    ret_raw = pd.concat([ret_transact, ret_midpoint, ret_industry], axis=1)
    ret_raw.columns = ['Transact. prices', 'Quote-midpoints', 'Industry portfolio']
    ret_raw.to_parquet(DATA_DIR / "derived" / "reversal_return_2010.parquet")

    ret_raw = load_reversal_return(data_dir=DATA_DIR)

    df_stat_A = summary_stats(ret_raw, reproduce=False)
    # df_stat_A = df_stat_A.style.format('{:.2f}',na_rep='')
    df_stat_A.to_parquet(DATA_DIR / "derived" / "Table_1A.parquet")


    # Reproduce Table 1A
    ret_industry_new = calc_reverse_strategy_industry(ff)
    ret_transact_new = calc_reverse_strategy_individual(dfcp)
    ret_midpoint_new = calc_reverse_strategy_individual(dfmid, 'quote_midpoint_return')

    ret_raw_new = pd.concat([ret_transact_new, ret_midpoint_new, ret_industry_new], axis=1)
    ret_raw_new.columns = ['Transact. prices', 'Quote-midpoints', 'Industry portfolio']
    ret_raw_new.to_parquet(DATA_DIR / "derived" / "reversal_return_2023.parquet")

    df_stat_A_new = summary_stats(ret_raw_new, reproduce=True)
    # df_stat_A_new = df_stat_A_new.style.format('{:.2f}',na_rep='')
    df_stat_A_new.to_parquet(DATA_DIR / "derived" / "Table_1A_reproduce.parquet")


    # Replicate Table 1B
    hedged_ret_transact = calc_hedged_return(ret_transact, reproduce=False)
    hedged_ret_midpoint = calc_hedged_return(ret_midpoint, reproduce=False)
    hedged_ret_industry = calc_hedged_return(ret_industry, reproduce=False)

    ret_hedged = pd.concat([hedged_ret_transact, hedged_ret_midpoint, hedged_ret_industry], axis=1)
    ret_hedged.columns = ['Hedged Transact. prices', 'Hedged Quote-midpoints', 'Hedged Industry portfolio']
    ret_hedged.to_parquet(DATA_DIR / "derived" / "reversal_return_hedged_2010.parquet")

    ret_hedged = load_reversal_return(data_dir=DATA_DIR, hedged=True, reproduce=False)
    
    df_stat_B = summary_stats(ret_hedged, reproduce=False)
    # df_stat_B = df_stat_B.style.format('{:.2f}',na_rep='')
    df_stat_B.to_parquet(DATA_DIR / "derived" / "Table_1B.parquet")


    # Reproduce Table 1B
    hedged_transact_new = calc_hedged_return(ret_transact_new, reproduce=True)
    hedged_midpoint_new = calc_hedged_return(ret_midpoint_new, reproduce=True)
    hedged_industry_new = calc_hedged_return(ret_industry_new, reproduce=True)

    ret_hedged_new = pd.concat([hedged_transact_new, hedged_midpoint_new, hedged_industry_new], axis=1)
    ret_hedged_new.columns = ['Hedged Transact. prices', 'Hedged Quote-midpoints', 'Hedged Industry portfolio']
    ret_hedged_new.to_parquet(DATA_DIR / "derived" / "reversal_return_hedged_2023.parquet")

    ret_hedged_new = load_reversal_return(data_dir=DATA_DIR, hedged=True, reproduce=True)
    
    df_stat_B_new = summary_stats(ret_hedged_new, reproduce=True)
    # df_stat_B_new = df_stat_B_new.style.format('{:.2f}',na_rep='')
    df_stat_B_new.to_parquet(DATA_DIR / "derived" / "Table_1B_reproduce.parquet")