#!/usr/bin/env python
# coding: utf-8

# # Final Project - Evaporating Liquidity
# 
# #### Ruilong Guo, Sifei Zhao, Zhiyuan Liu, Junhan Fu

# In this project, we replicate tables from the paper "_Evaporating Liquidity_" by Stefan Nagel using the Principals of Reproducible Analytical Pipelines (RAPs) learned in the class. 
# 
# The author demonstrates that short-term reversal strategy profits stem from liquidity provision, making them notably predictable by the VIX index. Additionally, the author discovered that these reversal strategies yield substantial returns not only for individual stocks but also for industry portfolios, particularly during high VIX periods.
# 
# Our replication is automated from end-to-end using Pydoit, formatted using the project template (blank_project) provided by professor Bejarano, which is based on the Cookiecutter Data Science template.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
import statsmodels.api as sm
import plotly.express as px

import load_CRSP_stock
import load_FF_industry
import load_vix

import clean_CRSP_stock
import calc_reversal_strategy
import additional_analysis
import regression
import regression_hac

DATA_DIR = config.DATA_DIR

import warnings
warnings.filterwarnings('ignore')


# ## Data Collection

# ### 1. Pull and load CRSP data from WRDS

# Using `load_CRSP_stock`, we pull and save CRSP daily stock data and indexes from WRDS (Wharton Research Data Services). 
# 
# The CRSP daily stock data is needed to construct individual portfolios based on Reversal strategy. The CRSP daily index data is needed to evaluate the performance of Reversal strategy portfolios.
# 
# Specifically:
# - we use query to pull data of stocks with share code 10 or 11, from NYSE, AMEX, and Nasdaq
# - pull one extra month of daily stock data for later data cleaning and processing

# #### 1) CRSP daily stock data

# In[ ]:


df_dsf = load_CRSP_stock.load_CRSP_daily_file(data_dir=DATA_DIR)
df_dsf.info()


# #### 2) CRSP daily indexes

# In[ ]:


df_msix = load_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)
df_msix.columns


# ### 2. Pull and load data from the Fama-French Data Library
# 

# 
# Using `load_FF_industry`, we pull and save 48 industry portfolio daily returns from the Fama/French Data Library. 
# 
# The industry portfolios are constructed by classifying stocks into 48 industries as in Fama and French (1997). The industry portfolio daily returns are needed to construct the industry portfolios based on Reversal strategy.

# In[ ]:


ff = load_FF_industry.load_FF_industry_portfolio_daily(data_dir=DATA_DIR)


# #### Average Value Weighted Daily Returns

# In[ ]:


ff[0].tail()


# ### 3. Pull and load VIX from the Fama-French Data Library
# 

# Using `load_vix`, we pull and save CBOE Volatility Index data from FRED. The data is used later in table replicatation.

# In[ ]:


vix = load_vix.load_vix(data_dir=DATA_DIR)


# ---

# ## Data Cleaning and Processing

# ### 1. Select the desired subsample of the data

# #### 1) For reversal strategy based on transaction prices calculated from daily closing prices

# 1. Adjust prc with a negative sign to be positive

# > Prc is the closing price or the negative bid/ask average for a trading day. If the closing price is not available on any given trading day, the number in the price field has a negative sign to indicate that it is a bid/ask average and not an actual closing price. Please note that in this field the negative sign is a symbol and that the value of the bid/ask average is not negative. If neither closing price nor bid/ask average is available on a date, prc is set to zero. 

# 2. Filter stocks with closing price (prc) of at least $1 on the last trading day of the previous calendar month

# > To enter into the sample, a stock needs to have share code 10 or 11. In addition, it must have a closing price of at least $1 on the last trading day of the previous calendar month.
# 

# 3. Select the CRSP stock data for a specific time period

# In[ ]:


def select_stocks_by_closing_prices(df):
    """
    Clean the CRSP stock data for strategy based on closing prices
    """

    # make sure 'prc' is positive (negative sign means using bid/ask average instead of closing price)
    df = clean_CRSP_stock.clean_prc_to_positive(df)

    # stocks must have a closing price of at least $1 on the last trading day of the previous calendar month
    df = clean_CRSP_stock.clean_1dollar_prc(df)

    # select time range of the data
    df = clean_CRSP_stock.clean_date(df, start_year=1998, end_year=2024)

    return df.reset_index()


# #### 2) For reversal strategy based on quote-midpoints calculated from averages of closing bid and ask quotes

# 1. Adjust prc with a negative sign to be positive
# 
# 2. Filter stocks with closing price (prc) of at least $1 on the last trading day of the previous calendar month
# 
# 3. Select stocks traded on NASDAQ
# 
# 4. Adjust bid and ask quotes for stock splits using CRSP adjustment factor

# > Reversal strategy returns based on transaction prices are calculated from daily closing prices, and the reversal strategy returns based on quote-midpoints are calculated from averages of closing bid and ask quotes, as reported in the CRSP daily returns file (for Nasdaq stocks only), adjusted for stock splits and dividends using the CRSP adjustment factors and dividend information.
# 

# 5. Filter stocks with ratio of bid to quote-midpoint not smaller than 0.5.
# 
# 6. Filter stocks with one-day return based on quote-midpoints minus the return based on closing prices not less than -50% and not higher than 100%. If a closing transaction price is not available, the quote-midpoint is used to calculate transaction-price returns.
# 
# 7. Select the CRSP stock data for a specific time period

# 
# > In a few instances, the closing bid and ask data for Nasdaq stocks on CRSP have some data recording errors, such as increases of bid or ask by a factor of 100 or digits that are cut off. 
# 
# > To screen out data recording errors of bid and ask data for Nasdaq stocks: require that the ratio of bid to quote-midpoint is not smaller than 0.5, and the one-day return based on quote-midpoints minus the return based on closing prices is not less than -50% and not higher than 100%. If a closing transaction price is not available, the quote-midpoint is used to calculate transaction-price returns.
# 

# In[ ]:


def select_stocks_by_quote_midpoints(df):
    """
    Clean the CRSP stock data for strategy based on quote-midpoints
    """

    # make sure 'prc' is positive (negative sign means using bid/ask average instead of closing price)
    df = clean_CRSP_stock.clean_prc_to_positive(df)

    # Nasdaq stocks only
    df = df[df['exchcd'] == 3]

    # adjusting the price using adjustment factors
    df['bid'] = df['bid'] / df['cfacpr']
    df['ask'] = df['ask'] / df['cfacpr']
    df['quote_midpoint'] = (df['bid'] + df['ask']) / 2

    # ratio of bid to quote-midpoint is not smaller than 0.5
    df = clean_CRSP_stock.clean_bid_quote_midpoint(df)

    # one-day return based on quote-midpoints minus the return based on closing prices is less than -50% and higher than 100%
    df = clean_CRSP_stock.clean_one_day_return(df)

    # stocks must have a closing price of at least $1 on the last trading day of the previous calendar month
    df = clean_CRSP_stock.clean_1dollar_prc(df)

    # select time range of the data
    df = clean_CRSP_stock.clean_date(df, start_year=1998, end_year=2024)

    return df.reset_index()


# ### 2. Load cleaned data

# In[ ]:


dfcp = clean_CRSP_stock.load_CRSP_closing_price(data_dir=DATA_DIR)
dfcp.info()


# In[ ]:


dfmid = clean_CRSP_stock.load_CRSP_midpoint(data_dir=DATA_DIR)
dfmid.info()


# ---

# ## Reversal Strategy 

# ### 1. Construct Reversal Strategy 

# Each day t, the reversal strategy returns are calculated as the average of returns from five reversal strategies that weight stocks proportional to the negative of market-adjusted returns on days t − 1, t − 2, ..., t − 5, with weights scaled to add up to $1 short and $1 long.
# 
# $$w_{it}^R = -\left( \frac{1}{2} \sum_{i=1}^{N} \left| R_{it-1} - R_{mt-1} \right| \right)^{-1} \left( R_{it-1} - R_{mt-1} \right)$$
# 
# - $R_{mt-1} = \frac{1}{N}\sum_{i=1}^N R_{it-1}$: the equal-weighted market return.

# In[ ]:


def calc_reverse_strategy_ret(df, type_col='industry', ret_col='ret'):
    df = df.copy()
    df['ret-avg'] = df.groupby('date')[ret_col].transform(lambda x: x - x.mean())
    df['w'] = df.groupby('date')['ret-avg'].transform(lambda x: - x / (0.5 * x.abs().sum()))

    for i in range(1, 6):
        df[f'w_lag_{i}'] = df.groupby(type_col)['w'].shift(i)

    df['rev_ret'] = (df['w_lag_1'] + df['w_lag_2'] + df['w_lag_3'] + df['w_lag_4'] + df['w_lag_5']) * df[ret_col] / 5

    return df.groupby('date')['rev_ret'].sum()


# To be specific, the returns of the reversal strategies are calculated as an overlay of the returns of five sub-strategies: One with portfolio weights conditioned on day t − 1 returns, one conditioned on day t − 2 data, ..., one conditioned on t − 5 data. Then we calculated the simple average of these five sub-strategies’ returns as the overall reversal strategy return.

# ### 2. Hedge against market factor risk

# Besides, we constructed the hedged reversal strategies after eliminating time-varying market factor exposure and then calculated the returns.
# 
# We first estimated a regression
# 
# $$L^R_t = β_0 + β_1f_t + β_2 (f_t × sgn(f_{t−1})) + ε_t$$
# - $f_t$ is the return on the CRSP value-weighted index
# - $L^R_t$ is the reversal strategy return
# 
# The time-varying beta is $β_{t−1} = β_1 + β_2 sgn(f_{t−1})$. Then hedged returns are calculated as $L^R_t − β_{t−1}f_t$.

# In[ ]:


def calc_hedged_return(ret):
    
    index = load_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)
    index = index.set_index('caldt')['vwretx']*100
    shifted_sign = index.shift(1).apply(lambda x: 1 if x > 0 else -1)
    index_shifted_sign = index * shifted_sign

    factor = pd.concat([index, index_shifted_sign], axis=1)

    beta = calc_reversal_strategy.calc_multiple_beta(factor, ret)
    time_varying_beta = beta[0] + beta[1] * shifted_sign

    hedged_ret = ret - time_varying_beta * index

    return hedged_ret


# ---

# ## Table Replication

# We replicate tables with data with time period from January 1998 to December 2010.

# ### __Table 1: Summary Statistics of Reversal Strategy Returns__

# Panel A presents statistics for the raw returns of these strategies, while Panel B shows similar statistics for hedged returns, which are obtained by eliminating conditional market factor exposure.

# #### Panel A. Raw returns

# Reversal strategy returns are calculated both with transaction-price returns, quote-midpoint returns, and industry returns. In each case, the portfolio weights are calculated with the same type of return (on days t − 1 to t − 5) as the type used to calculate portfolio returns (on day t).
# 
# - Transaction-price returns are calculated from daily CRSP closing transaction prices (for NYSE, AMEX, and Nasdaq stocks). 
# - Quote-midpoint returns are calculated from bid-ask midpoints of daily CRSP closing quotes (with Nasdaq stocks only). 
# - The industry returns pulled from Fama and French are calculated with transaction prices. 

# In[ ]:


ret_raw = calc_reversal_strategy.load_reversal_return(data_dir=DATA_DIR)
ret_raw.describe()


# In[ ]:


ret_raw.plot(subplots=True, figsize=(8, 8));


# #### Replication of Table 1A

# In[ ]:


df_stat_A = calc_reversal_strategy.summary_stats(ret_raw)
df_stat_A


# #### Panel B. Returns hedged for conditional market

# Also, we calculated the returns of hedged reversal strategies after eliminating time-varying market factor exposure.

# In[ ]:


ret_hedged = calc_reversal_strategy.load_reversal_return(data_dir=DATA_DIR, hedged=True)
ret_hedged.describe()


# In[ ]:


ret_hedged.plot(subplots=True, figsize=(8, 8));


# #### Replication of Table 1B

# In[ ]:


df_stat_B = calc_reversal_strategy.summary_stats(ret_hedged)
df_stat_B


# ### __Table 2: Predicting Reversal Strategy Returns with VIX__

# As above, Table 2 is also replicated with data with time period from January 1998 to December 2010.

# To replicate the prediction of returns from liquidity provision with VIX, we ran a regression of the form
# 
# $$ L^R_t =a+bVIX_t−5 +c'g_{t−5} +e_t$$

# - $L^R_t$ is the reversal strategy return on day t.
# 
# - VIX is lagged by five days to account for the fact that the portfolio weights of the day t reversal strategy are conditioned on returns from days t − 1 to t − 5. In these regressions, the VIX is normalized to a daily volatility measure by dividing it by $\sqrt{250}$. 
# 
# - To control for effects of the institutional changes associated with decimalization, the control variable vector $g_{t−5}$ includes a dummy variable that takes a value of one prior to decimalization (April 9, 2001) and a value of zero thereafter. $R_M$ is also included in the control variable vector $g_{t−5}$, which is the lagged four-week return on the value-weighted CRSP index up until the end of day t − 5 to capture the dependence of reversal strategy profits on lagged market returns.

# #### 1. Data preparation

# $L^R_t$: Reversal strategy returns

# In[ ]:


reversal_ret = calc_reversal_strategy.load_reversal_return()
reversal_ret.columns = ['trade', 'quote', 'industry']


# VIX: the CBOE S&P500 implied volatility index
# 

# In[ ]:


vix = load_vix.load_vix()
vix.columns = ['VIX']


# $R_M$: the lagged four-week return on the value-weighted CRSP index
# 

# In[ ]:


rm = load_CRSP_stock.load_CRSP_index_files()
rm = rm.set_index('caldt')[['vwretx']]
rm.columns = ['$R_M$']


# In the daily regressions, the dependent variable is the reversal strategy return on day t (in percent), and the predictor variables are measured at the end of day t − 5. 

# In[ ]:


data = regression_hac.prepare_data(reversal_ret, vix, rm)
data


# In the monthly regressions, the dependent variable is the monthly average of daily reversal strategy returns, and the predictor variables are measured five days before the end of the month preceding the return measurement month.

# In[ ]:


data_m = regression_hac.prepare_data(reversal_ret, vix, rm, to_monthly=True)
data_m


# #### 2. Regression with HAC standard errors
# 

# In[ ]:


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


# #### 3. Table 2 Replication

# In[ ]:


table = regression_hac.generate_table(data, data_m)
table


# Note: Newey-West HAC standard errors (with 20 lags for daily data and 3 lags for monthly data) are reported in parentheses.

# ---

# ## Table Reproduction

# Here, we reproduce tables with updated data until December 2023.

# ### Table 1: Summary Statistics of Reversal Strategy Returns

# #### Panel A. Raw returns

# In[ ]:


ret_raw_new = calc_reversal_strategy.load_reversal_return(data_dir=DATA_DIR, reproduce=True)
ret_raw_new.describe()


# In[ ]:


ret_raw_new.plot(subplots=True, figsize=(8, 8));


# #### Reproduction of Table 1A

# In[ ]:


df_stat_A_new = calc_reversal_strategy.summary_stats(ret_raw_new, reproduce=True)
df_stat_A_new


# #### Panel B. Returns hedged for conditional market

# In[ ]:


ret_hedged_new = calc_reversal_strategy.load_reversal_return(data_dir=DATA_DIR, hedged=True, reproduce=True)
ret_hedged_new.describe()


# In[ ]:


ret_hedged_new.plot(subplots=True, figsize=(8, 8));


# #### Reproduction of Table 1B

# In[ ]:


df_stat_B_new = calc_reversal_strategy.summary_stats(ret_hedged_new, reproduce=True)
df_stat_B_new


# ### Table 2: Predicting Reversal Strategy Returns with VIX

# #### Table 2 Reproduction

# In[ ]:


reversal_ret_new = calc_reversal_strategy.load_reversal_return(reproduce=True)
reversal_ret_new.columns = ['trade', 'quote', 'industry']


# In[ ]:


data_new = regression_hac.prepare_data(reversal_ret_new, vix, rm)
data_m_new = regression_hac.prepare_data(reversal_ret_new, vix, rm, to_monthly=True)


# In[ ]:


table = regression_hac.generate_table(data_new, data_m_new)
table


# Note: Newey-West HAC standard errors (with 20 lags for daily data and 3 lags for monthly data) are reported in parentheses.

# ---

# ## Analysis outside of replication

# ### 1. Performance of Reversal Strategies

# In[ ]:


ret_raw = calc_reversal_strategy.load_reversal_return(data_dir=DATA_DIR)

index = pd.read_parquet(DATA_DIR / "pulled" / "CRSP_DSIX.parquet")
index = index.set_index('caldt')['vwretx']*100

strategies = pd.concat([ret_raw, index], axis=1)
strategies.columns = ['Transact. prices','Quote-midpoints','Industry portfolio', 'CRSP Value Weighted Index']


# In[ ]:


performance_matrix = additional_analysis.performance_summary(strategies, 252)
performance_matrix


# Apart from the original statistical analysis of reversal strategy provided by the paper, we create a new version of performance matrix which includes VaR(0.05), CVaR(0.05), max drawdown, and other drawdown-based strategy perfomance, and we also add CRSP value weighted index as the benchmark to evaluate the performance of reversal strategies. 
# 
# Compared to the CRSP value weighted index, the reversal strategy based on individual stocks tends to have much higher annualized mean return and lower annualized volatility, which cause a way higher annualized sharpe ratio. The mean return of industry reversal strategy is a little bit lower than the banchmark, but it has lower volatility with higher sharpe ratio.
# 
# With regard to max drawdown, the transact price based individual reversal strategy is the best(-4.38%) among all the reversal strategies(quote-midpoints: -7.70%, industry: -13.90%) and the CRSP index(-57.18%). That strategy dropped form the peak on 2009-10-22 after the period of financial crisis. And it only used 7 days to recover the lose since the peak, while the industry reversal strategy took 433 days to recover and CRSP value weighted index didn't recover to the peak.

# ### 2. Reversal Strategy vs VIX

# In[ ]:


vix = load_vix.load_vix(data_dir=DATA_DIR)
reversal_strategy = ret_raw.join(vix, how='left')
reversal_strategy['Transact. prices'] = reversal_strategy['Transact. prices'].rolling(window=62).mean()


# In[ ]:


# Plot reversal strategy and VIX
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(reversal_strategy.index, reversal_strategy['Transact. prices'], color='brown')
ax1.set_ylabel('Average return per day')
plt.legend(['Average return per day'])

ax2 = ax1.twinx()
ax2.plot(reversal_strategy.index, reversal_strategy['VIXCLS'])
ax2.set_ylabel('VIX')
plt.legend(['VIX'])

ax1.set_xlabel('Date')
plt.show()


# This figure shows the three-month moving average return of the reversal strategy and VIX index across 1998 to 2010. The blue curve(VIX index) has a pre-trend of the red curve(3-month MA return of reversal strategy), which presents a key finding of the paper that the VIX index has a power to predict the reversal strategy return.
# 
# During the LTCM crisis in 1998 and  Nasdaq decline in 2000, the reversal strategy return increased with VIX increasing. From then until 2007, returns declined steadily to less than 0.2% per day, but during the financial crisis, they surged, surpassing levels seen during the LTCM crisis. The figure illustrates a strong correlation between the time variation in the reversal strategy's return and the VIX index. Since the financial crisis began in 2007, the returns of the reversal strategy and the VIX have closely tracked each other.
# 
