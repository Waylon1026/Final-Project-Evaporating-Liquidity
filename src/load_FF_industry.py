"""
This module pulls and saves 48 industry portfolio daily returns from the Fama/French Data Library.
The industry portfolios are constructed by classifying stocks into 48 industries as in Fama and French (1997).
The data is needed to evaluate performance of the industry portfolios used in Reversal strategy.
"""

import pandas as pd
import pandas_datareader
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
START_DATE = config.START_DATE
END_DATE = config.END_DATE
# START_DATE="1998-01-01"
# END_DATE="2010-12-31"


def pull_FF_industry_portfolio_data(start=START_DATE, end=END_DATE):
    """
    Pull 48 industry portfolio daily returns
    from the Fama/French Data Library from a specified start date to end date.
    """

    df = pandas_datareader.data.DataReader("48_Industry_Portfolios_daily", "famafrench", start=start, end=end)
    return df


def load_FF_industry_portfolio_data(data_dir=DATA_DIR):
    """
    Load 48 industry portfolio daily returns

    ff[0]: Average Value Weighted Returns
    ff[1]: Average Equal Weighted Returns
    """
    path = Path(data_dir) / "pulled" / "48_Industry_Portfolios_daily.parquet"
    ff = pd.read_parquet(path)
    return ff


def demo():
    ff = load_FF_industry_portfolio_data(data_dir=DATA_DIR)


if __name__ == "__main__":
    ff = pull_FF_industry_portfolio_data()
    ff.to_parquet(DATA_DIR / "pulled" / "48_Industry_Portfolios_daily.parquet")
