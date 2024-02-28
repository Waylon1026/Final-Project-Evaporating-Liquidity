import pandas as pd
import numpy as np
import config

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

### Define several functions to clean the CRSP stock data ###




######################

# take the last trading day of the previous calendar month

# filter stocks with closing price (prc) of at least $1 on the last trading day of the previous calendar month



######################

# calculate the ratio of bid to quote-midpoint

# calculate one-day return based on quote-midpoints minus the return based on closing prices

# filter stocks with ratio of bid to quote-midpoint not smaller than 0.5
# and with the one-day return based on quote-midpoints minus the return based on closing prices 
# is not less than -50% and not higher than 100%



######################

# calculate returns based on transaction prices (using daily closing prices)
# or directly use ret (with div) / retx (without div)
    # ret / retx: missing returns are indicated by a value of -66.0,-77.0,-88.0, or -99.0


# calculate returns based on quote-midpoints (using averages of closing bid and ask quotes)




######################
# Helper function

## Time period: 1998 - 2010
def clean_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'].dt.year >= 1998) & (df['date'].dt.year <= 2010)]
    return df 

## stocks must have a closing price of at least $1 on the last trading day of the previous calendar month
## Junhan's version: For a given stock at day t, if the closing price of the stock at last day of previous month is less than 1, then this row will be removed from the dataset.
## (not sure if this is the correct way to do it)

def clean_1dollar_prc(df):
    df['period'] = df['date'].dt.to_period('M')
    df_month = df.copy()
    df_month = df_month.groupby(['permno', 'period'], as_index=False).last()
    df_month = df_month[df_month['prc'] < 1]
    df_month['period'] = df_month['period'].apply(lambda x: x + 1)

    df['key'] = df['permno'].astype(str) + df['period'].astype(str)
    df_month['key'] = df_month['permno'].astype(str) + df_month['period'].astype(str)
    df = df[~df['key'].isin(df_month['key'])].drop(columns=['key', 'period'])
    
    return df


## ratio of bid to quote-midpoint is not smaller than 0.5
def clean_bid_quote_midpoint(df):
    df = df[(df['bid'] / df['quote_midpoint'] >= 0.5)]
    return df


## one-day return based on quote-midpoints minus the return based on closing prices is not less than -50% and not higher than 100%
## If a closing transaction price is not available, the quote-midpoint is used to calculate transaction-price returns.
def clean_one_day_return(df):
    df['prc'] = np.where(df['prc'].isnull(), df['quote_midpoint'], df['prc'])
    df['transaction_price_return'] = df['prc'].pct_change()
    df['quote_midpoint_return'] = df['quote_midpoint'].pct_change()
    df = df[(df['quote_midpoint_return'] - df['transaction_price_return'] >= -0.5) & (df['quote_midpoint_return'] - df['transaction_price_return'] <= 1)]
    return df.drop(columns=['transaction_price_return', 'quote_midpoint_return'])


######################



######################
# Clean the CRSP stock data for strategy based on closing prices
def select_stocks_by_closing_prices(df):

    # time range
    # df = clean_date(df)

    # stocks must have a closing price of at least $1 on the last trading day of the previous calendar month
    df = clean_1dollar_prc(df)

    return df.reset_index()

######################



######################
# Clean the CRSP stock data for strategy based on quote-midpoints
def select_stocks_by_quote_midpoints(df):

    # time range
    # df = clean_date(df)

    # Nasdaq stocks only
    df = df[df['exchcd'] == 3]

    # stocks must have a closing price of at least $1 on the last trading day of the previous calendar month
    df = clean_1dollar_prc(df)

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

    return df.reset_index()

######################



def _demo():
    df_dsf = load_CRSP_stock.load_CRSP_daily_file(data_dir=DATA_DIR)


if __name__ == "__main__":
    df_dsf = load_CRSP_stock.load_CRSP_daily_file(data_dir=DATA_DIR)

    df_closing_prices = select_stocks_by_closing_prices(df_dsf)
    df_closing_prices.to_parquet(DATA_DIR / "pulled" / "CRSP_closing_price.parquet")


    df_midpoint = select_stocks_by_quote_midpoints(df_dsf)
    df_midpoint.to_parquet(DATA_DIR / "pulled" / "CRSP_midpoint.parquet")