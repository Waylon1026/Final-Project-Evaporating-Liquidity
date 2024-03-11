"""
This file is used to test the loaded data.
"""

import pandas as pd
import pytest

import config
import load_CRSP_stock
import load_FF_industry
import load_vix

DATA_DIR = config.DATA_DIR


def test_load_CRSP_stock_data():
    df = load_CRSP_stock.load_CRSP_daily_file()
    # Test if the function returns a pandas DataFrame
    assert isinstance(df, pd.DataFrame)

    # Test if the DataFrame has the expected columns
    expected_columns = ['date', 'permno', 'permco', 'exchcd', 'prc', 
                        'bid', 'ask', 'shrout', 'cfacpr', 'cfacshr', 'ret', 'retx']
    assert all(col in df.columns for col in expected_columns)

    # Test if the function raises an error when given an invalid data directory
    with pytest.raises(FileNotFoundError):
        load_CRSP_stock.load_CRSP_daily_file(data_dir="invalid_directory")

    # # Test if the average annualized growth rate is close to 3.08%
    # ave_annualized_growth = 4 * 100 * df.loc['1913-01-01': '2023-09-01', 'GDPC1'].dropna().pct_change().mean()
    # assert abs(ave_annualized_growth - 3.08) < 0.1
