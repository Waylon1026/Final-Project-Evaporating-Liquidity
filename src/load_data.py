'''
@Author: Zhiyuan Liu
@Date: 2024-02-21
'''

import pandas as pd
import pandas_datareader
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)


def load_vix_from_fred(start="1998-01-01", end="2010-12-31"):
    """
    Load VIX data from FRED
    """
    df = pandas_datareader.get_data_fred("VIXCLS", start=start, end=end)
    return df

def load_crsp_stock_daily():
    """
    Load CRSP daily stock return from WRDS
    """
    pass


if __name__ == "__main__":
    df = load_vix_from_fred()
    df.to_parquet(DATA_DIR / "vix.parquet")