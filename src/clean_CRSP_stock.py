"""
This module cleans the CRSP daily stock data step-by-step using several helper functions.
The cleaned data is needed to construct the individual portfolios used in Reversal strategy.
"""

import pandas as pd
import numpy as np
import config
from pathlib import Path

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR
WRDS_USERNAME = config.WRDS_USERNAME

import misc_tools
import load_CRSP_stock


"""
Reversal strategy returns based on transaction prices are calculated from daily closing prices, 
and the reversal strategy returns based on quote-midpoints are calculated from averages of closing bid and ask quotes, 
as reported in the CRSP daily returns file (for Nasdaq stocks only), 
adjusted for stock splits and dividends using the CRSP adjustment factors and dividend information.
"""

"""
To into the sample, stocks must have a closing price of at least $1 on the last trading day of the previous calendar month

To screen out data recording errors of bid and ask data for Nasdaq stocks:
require that the ratio of bid to quote-midpoint is not smaller than 0.5, 
and the one-day return based on quote-midpoints minus the return based on closing prices 
is not less than -50% and not higher than 100%.

If a closing transaction price is not available, 
the quote-midpoint is used to calculate transaction-price returns. (already considered in prc??)
"""


### Helper functions to clean the CRSP stock data ###


def clean_date(df, start_year=1998, end_year=2024):
    """
    Select the CRSP stock data for a specific time period
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
    return df 


def clean_prc_to_positive(df):
    """
    Adjust prc with a negative sign to be positive
    
    A negative sign means prc is a bid/ask average instead of a closing price
    
    # Prc is the closing price or the negative bid/ask average for a trading day. 
    # If the closing price is not available on any given trading day, 
    # the number in the price field has a negative sign to indicate that it is a bid/ask average and not an actual closing price. 
    # Please note that in this field the negative sign is a symbol and that the value of the bid/ask average is not negative.
    # If neither closing price nor bid/ask average is available on a date, prc is set to zero. 
    """
    df['prc'] = np.abs(df['prc'])
    return df


## stocks must have a closing price of at least $1 on the last trading day of the previous calendar month
## Junhan's version: For a given stock at day t, if the closing price of the stock at last day of previous month is less than 1, then this row will be removed from the dataset.
## update: Exclude the stock from the sample, not just the row (period)

def clean_1dollar_prc(df):
    """
    Filter stocks with closing price (prc) of at least $1 
    on the last trading day of the previous calendar month
    """
    df['period'] = df['date'].dt.to_period('M')
    df_month = df.copy()
    df_month = df_month.groupby(['permno', 'period'], as_index=False).last()
    df_month = df_month[df_month['prc'] < 1]
    df_month['period'] = df_month['period'].apply(lambda x: x + 1)

    df['key'] = df['permno'].astype(str) #+ df['period'].astype(str)
    df_month['key'] = df_month['permno'].astype(str) #+ df_month['period'].astype(str)
    df = df[~df['key'].isin(df_month['key'])].drop(columns=['key', 'period'])
    
    return df


def clean_bid_quote_midpoint(df):
    """
    Filter stocks with ratio of bid to quote-midpoint not smaller than 0.5
    """
    df = df[(df['bid'] / df['quote_midpoint'] >= 0.5)]
    return df


def clean_one_day_return(df):
    """
    Filter stocks with one-day return based on quote-midpoints 
    minus the return based on closing prices not less than -50% and not higher than 100%

    If a closing transaction price is not available, 
    the quote-midpoint is used to calculate transaction-price returns.
    """
    df['prc'] = np.where(df['prc'].isnull(), df['quote_midpoint'], df['prc'])

    df['transaction_price_return'] = df.groupby('permno')['prc'].pct_change()
    df['quote_midpoint_return'] = df.groupby('permno')['quote_midpoint'].pct_change()
    df = df[(df['quote_midpoint_return'] - df['transaction_price_return'] >= -0.5) & (df['quote_midpoint_return'] - df['transaction_price_return'] <= 1)]

    return df.drop(columns=['transaction_price_return', 'quote_midpoint_return'])


########################################################################################

### Clean the data using helper functions ###


def select_stocks_by_closing_prices(df):
    """
    Clean the CRSP stock data for strategy based on closing prices
    """

    # make sure 'prc' is positive
    # negative sign means using bid/ask average instead of closing price
    df = clean_prc_to_positive(df)

    # stocks must have a closing price of at least $1 on the last trading day of the previous calendar month
    df = clean_1dollar_prc(df)

    # select time range of the data
    df = clean_date(df, start_year=1998, end_year=2024)

    return df.reset_index()


def select_stocks_by_quote_midpoints(df):
    """
    Clean the CRSP stock data for strategy based on quote-midpoints
    """

    # make sure 'prc' is positive (negative sign means using bid/ask average instead of closing price)
    df = clean_prc_to_positive(df)

    # Nasdaq stocks only
    df = df[df['exchcd'] == 3]

    # calculate mid-quote

    ## Q: whether to use the adjustment factors or not???
    # # adjusting the price using adjustment factors
    # df['bid'] = df['bid'] / df['cfacpr']
    # df['ask'] = df['ask'] / df['cfacpr']
    df['quote_midpoint'] = (df['bid'] + df['ask']) / 2

    # clean the sample: ratio of bid to quote-midpoint is not smaller than 0.5
    df = clean_bid_quote_midpoint(df)

    # clean the sample: one-day return based on quote-midpoints minus the return based on closing prices is less than -50% and higher than 100%
    df = clean_one_day_return(df)

    # stocks must have a closing price of at least $1 on the last trading day of the previous calendar month
    df = clean_1dollar_prc(df)

    # select time range of the data
    df = clean_date(df, start_year=1998, end_year=2024)

    return df.reset_index()


########################################################################################

### Functions to load the dataset ###


def load_CRSP_closing_price(data_dir=DATA_DIR):
    """
    Load cleaned CRSP stock data for strategy based on closing prices
    """
    path = Path(data_dir) / "pulled" / "CRSP_closing_price.parquet"
    df = pd.read_parquet(path)
    return df


def load_CRSP_midpoint(data_dir=DATA_DIR):
    """
    Load cleaned CRSP stock data for strategy based on quote-midpoints
    """
    path = Path(data_dir) / "pulled" / "CRSP_midpoint.parquet"
    df = pd.read_parquet(path)
    return df



def _demo():
    dfcp = load_CRSP_closing_price(data_dir=DATA_DIR)
    dfmid = load_CRSP_midpoint(data_dir=DATA_DIR)


if __name__ == "__main__":
    df_dsf = load_CRSP_stock.load_CRSP_daily_file(data_dir=DATA_DIR)

    df_closing_prices = select_stocks_by_closing_prices(df_dsf)
    df_closing_prices.to_parquet(DATA_DIR / "pulled" / "CRSP_closing_price.parquet")

    df_midpoint = select_stocks_by_quote_midpoints(df_dsf)
    df_midpoint.to_parquet(DATA_DIR / "pulled" / "CRSP_midpoint.parquet")