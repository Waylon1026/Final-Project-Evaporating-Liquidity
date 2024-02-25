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

#####################

crsp_pulled = load_CRSP_stock.pull_CRSP_daily_file(wrds_username=WRDS_USERNAME)

crsp = crsp_pulled.copy()


# groupby permno, date
crsp = crsp.groupby(['permno', 'date']).first()

# drop permco
crsp = crsp.drop(columns=['permco'])

# calculate quote-midpoint (average of closing bid and ask quotes)
crsp['quote_midpoint'] = (crsp['bid'] + crsp['ask']) / 2



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



def _demo():
    df_dsf = load_CRSP_stock.load_CRSP_daily_file(data_dir=DATA_DIR)


if __name__ == "__main__":
    pass