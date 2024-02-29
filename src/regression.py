import pandas as pd
import numpy as np
import config
import statsmodels.api as sm


OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR
WRDS_USERNAME = config.WRDS_USERNAME


import misc_tools
import load_CRSP_stock
import load_FF_industry

"""
VIX
VIX is lagged by five days to account for the fact that the portfolio weights of the day t reversal strategy are conditioned on returns from days t − 1 to t − 5. 
In these regressions, the VIX is normalized to a daily volatility measure by dividing it by √250
"""

def build_vix_factor(vix):
    """
    Build VIX factor
    """
    vix["VIX"] = vix["VIXCLS"].shift(5)
    vix["VIX"] = vix["VIX"] / np.sqrt(250)
    return vix

"""
g t-5
Control variable vector gt−5 includes a dummy variable that takes a value of one prior to decimalization (April 9, 2001) and a value of zero thereafter.
"""

def build_dummy(df):
    """
    Build dummy variable
    """
    df["dummy"] = (df.index <= "2001-04-09").astype(int)
    return df



"""
RM
RM , the lagged four-week return on the value-weighted CRSP index up until the end of day t − 5 to capture the dependence of reversal strategy profits on lagged market returns 
"""

def build_rm_factor(df):
    """
    Build RM factor
    """
    df_dsix = load_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)
    df_dsix["vwretd"] = df_dsix["vwretd"].shift(5)
    df_dsix['RM'] = (1 + df_dsix['vwretd']).rolling(20).apply(np.prod, raw=True) - 1
    df = pd.merge(df, df_dsix[["RM"]], left_index=True, right_on='caldt')
    return df
    




"""
Assume now we have: 1. reversal strategy returns based on transaction prices, 2. reversal strategy returns based on quote-midpoints, 3. FF industry daily return
Three dataframe have the same structure: index = date, column = daily return

Build a linear regression function where strategy returns are the dependent variable and VIX, RM, and dummy are the independent variables.
"""

def daily_return_regression(str_return, vix, use_dummy = False, use_rm = False):
    """
    Linear regression of daily return
    """
    # merge strategy return with VIX
    df = pd.merge(str_return, vix, left_index=True, right_index=True)
    df = build_vix_factor(df)

    # use dummy variable
    if use_dummy:
        df = build_dummy(df)
    # use RM
    if use_rm:
        df = build_rm_factor(df)

    # regression
    X = df[["VIX"]]
    if use_dummy:
        X["dummy"] = df["dummy"]
    if use_rm:
        X["RM"] = df["RM"]

    X = sm.add_constant(X)
    y = df[str_return.columns[0]]
    model = sm.OLS(y, X).fit()
    return model.params, model.bse, model.rsquared



"""
Monthly Regression
monthly regressions in which the daily reversal strategy returns are averaged within each calendar month, 
and the predictor variables are measured five days before the end of the prior month.
"""
def monthly_return_regression(str_return, vix):
    """
    Monthly regression of daily return
    """
    # merge strategy return with VIX
    df = pd.merge(str_return, vix, left_index=True, right_index=True)
    df = build_vix_factor(df)
    df = build_dummy(df)
    df = build_rm_factor(df)

    # group by year-month, and take the last 5th day values of the month, shift by 1 to be the predictor of the next month
    df = df.reset_index(names=['date'])
    df = df.groupby(df['date'].dt.to_period('M')).apply(lambda x: x.iloc[-5])
    df = df.shift(1)

    # regression
    X = df[["VIX", "dummy", "RM"]]
    X = sm.add_constant(X)
    y = df[str_return.columns[0]].resample("M").mean()
    model = sm.OLS(y, X).fit()
    return model.params, model.bse, model.rsquared






def _demo():
    pass

if __name__ == "__main__":
    
    # load reversal strategy returns

    # daily regression 3 times: vix, vix+dummy, vix+dummy+rm

    # monthly regression 1 time: vix+dummy+rm

    pass

  