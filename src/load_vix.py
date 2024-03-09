'''
@Author: Zhiyuan Liu
@Date: 2024-02-21
'''

import pandas as pd
import pandas_datareader
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
START_DATE = config.START_DATE
END_DATE = config.END_DATE


def pull_vix_from_fred(start=START_DATE, end=END_DATE):
    """
    pull VIX data from FRED
    """
    df = pandas_datareader.get_data_fred("VIXCLS", start=start, end=end)
    return df


def load_vix(data_dir=DATA_DIR):
    """
    Load VIX data from local file
    """
    path = Path(data_dir) / "pulled" / "vix.parquet"
    return pd.read_parquet(path)



if __name__ == "__main__":
    df = pull_vix_from_fred()
    df.to_parquet(DATA_DIR / "pulled" / "vix.parquet")
