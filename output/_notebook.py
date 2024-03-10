#!/usr/bin/env python
# coding: utf-8

# # Final Project - Evaporating Liquidity

# In this project, we replicate tables from the paper "_Evaporating Liquidity_" by Stefan Nagel using the Principals of Reproducible Analytical Pipelines (RAPs) learned in the class. 
# 
# Our replication is automated from end-to-end using Pydoit, formatted using the project template (blank_project) provided by professor Bejarano, which is based on the Cookiecutter Data Science template.

# In[ ]:


import pandas as pd

import config

import load_CRSP_stock
import load_FF_industry
import load_vix

import clean_CRSP_stock
import calc_reversal_strategy
# import hedged_reversal_strategy

DATA_DIR= config.DATA_DIR

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

# #### CRSP daily stock data

# In[ ]:


df_dsf = load_CRSP_stock.load_CRSP_daily_file(data_dir=DATA_DIR)
df_dsf.info()


# ### CRSP daily indexes

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

# 
# Using `load_vix`, we pull and save CBOE Volatility Index data from FRED. The data is used later in table replicatation.

# In[ ]:


vix = load_vix.load_vix(data_dir=DATA_DIR)


# ## Data Cleaning and Processing

# ### Select the desired subsample of the data

# #### For reversal strategy based on transaction prices calculated from daily closing prices

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


# #### For reversal strategy based on quote-midpoints calculated from averages of closing bid and ask quotes

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


# ### Load cleaned data

# In[ ]:


dfcp = clean_CRSP_stock.load_CRSP_closing_price(data_dir=DATA_DIR)
dfcp.info()


# In[ ]:


dfmid = clean_CRSP_stock.load_CRSP_midpoint(data_dir=DATA_DIR)
dfmid.info()


# ## Reversal Strategy 

# ### Construct Reversal Strategy 

# Each day t, the reversal strategy returns are calculated as the average of returns from five reversal strategies that weight stocks proportional to the negative of market-adjusted returns on days t − 1, t − 2, ..., t − 5, with weights scaled to add up to $1 short and $1 long.

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

# ### Hedge against market factor risk

# In[ ]:





# ## Table Replication

# We replicate tables with data with time period from January 1998 to December 2010.

# ### Table 1: Summary Statistics of Reversal Strategy Returns

# Panel A presents statistics for the raw returns of these strategies, while Panel B shows similar statistics for hedged returns, which are obtained by eliminating conditional market factor exposure.

# #### A. Raw returns

# Reversal strategy returns are calculated both with transaction-price returns, quote-midpoint returns, and industry returns. In each case, the portfolio weights are calculated with the same type of return (on days t − 1 to t − 5) as the type used to calculate portfolio returns (on day t).
# 
# - Transaction-price returns are calculated from daily CRSP closing transaction prices (for NYSE, AMEX, and Nasdaq stocks). 
# - Quote-midpoint returns are calculated from bid-ask midpoints of daily CRSP closing quotes (with Nasdaq stocks only). 
# - The industry returns pulled from Fama and French are calculated with transaction prices. 

# In[ ]:


ret_raw = calc_reversal_strategy.load_reversal_return(data_dir=DATA_DIR)
ret_raw


# In[ ]:


df_stat_A = calc_reversal_strategy.summary_stats(ret_raw)
df_stat_A = df_stat_A.style.format('{:.2f}',na_rep='')
df_stat_A


# #### B. Hedged returns

# In[ ]:


ret_hedged = calc_reversal_strategy.load_reversal_return(data_dir=DATA_DIR, hedged=True)
ret_hedged


# In[ ]:


df_stat_B = calc_reversal_strategy.summary_stats(ret_hedged)
df_stat_B = df_stat_B.style.format('{:.2f}',na_rep='')
df_stat_B


# ### Table 2: Predicting Reversal Strategy Returns with VIX

# In[ ]:





# ## Table Reproduction

# Here, we reproduce tables with updated data.

# ### Table 1: Summary Statistics of Reversal Strategy Returns

# In[ ]:





# ### Table 2: Predicting Reversal Strategy Returns with VIX

# In[ ]:





# ## Analysis outside of replication

# In[ ]:




