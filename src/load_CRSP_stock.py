"""
This module pulls and saves CRSP daily stock data and indexes from WRDS.
The data is needed to construct portfolios used in Reversal strategy.
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd
import wrds

import config

DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME
START_DATE = config.START_DATE
END_DATE = config.END_DATE


def pull_CRSP_daily_file(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """
    Pulls daily CRSP stock data from a specified start date to end date.

    Uses SQL query to pull data of stocks with share code 10 or 11, 
    from NYSE, AMEX, and Nasdaq.
    """
    # pull one extra month of data for cleaning the data
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date = start_date - relativedelta(months=1)
    start_date = start_date.strftime("%Y-%m-%d")

    query = f"""
    SELECT 
        date,
        dsf.permno, dsf.permco, exchcd, 
        prc, bid, ask, shrout, cfacpr, cfacshr,
        ret, retx
    FROM crsp.dsf AS dsf
    LEFT JOIN 
        crsp.msenames as msenames
    ON 
        dsf.permno = msenames.permno AND
        msenames.namedt <= dsf.date AND
        dsf.date <= msenames.nameendt
    WHERE 
        dsf.date BETWEEN '{start_date}' AND '{end_date}' AND 
        msenames.shrcd IN (10, 11) AND
        msenames.exchcd BETWEEN 1 AND 3
    """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     df = db.raw_sql(
    #         query, date_cols=["date", "namedt", "nameendt", "dlstdt"]
    #     )
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(
        query, date_cols=["date"]
    )
    db.close()

    df = df.loc[:, ~df.columns.duplicated()]
    df["shrout"] = df["shrout"] * 1000

    return df


def load_CRSP_daily_file(data_dir=DATA_DIR):
    """
    Load daily CRSP stock data
    """
    path = Path(data_dir) / "pulled" / "CRSP_stock.parquet"
    crsp = pd.read_parquet(path)
    return crsp



def pull_CRSP_index_files(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    # Pull index files
    query = f"""
        SELECT * 
        FROM crsp.dsix
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     df = db.raw_sql(query, date_cols=["month", "caldt"])
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()
    return df


def load_CRSP_index_files(data_dir=DATA_DIR):
    path = Path(data_dir) / "pulled" / f"CRSP_DSIX.parquet"
    df = pd.read_parquet(path)
    return df


def demo():
    df_dsf = load_CRSP_daily_file(data_dir=DATA_DIR)
    df_msix = load_CRSP_index_files(data_dir=DATA_DIR)


if __name__ == "__main__":
    df_dsf = pull_CRSP_daily_file(wrds_username=WRDS_USERNAME)
    df_dsf.to_parquet(DATA_DIR / "pulled" / "CRSP_stock.parquet")

    df_msix = pull_CRSP_index_files(start_date=START_DATE, end_date=END_DATE)
    path = Path(DATA_DIR) / "pulled" / f"CRSP_DSIX.parquet"
    df_msix.to_parquet(path)