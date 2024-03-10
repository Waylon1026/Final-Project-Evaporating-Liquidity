import pandas as pd
import numpy as np
import config
import calc_reversal_strategy

DATA_DIR = config.DATA_DIR

"""
Test whether the table is formatted correctly
- column and index names
- dimension size
"""
def test_format(df_raw, df_hedge):
    expected_index = ['Mean return(% per day)', 'Std.dev.(% per day)','Skewness', 'Kurtosis', 'Worst day return(%)',
                        'Worst 3-month return(%)', 'Beta', 'Annualized Sharpe Ratio']
    expected_columns = ['Transact. prices', 'Quote-midpoints', 'Industry portfolio']
    expected_columns_hedge = ['Hedged Transact. prices', 'Hedged Quote-midpoints', 'Hedged Industry portfolio']
    assert df_raw.shape == (8, 3)
    assert df_raw.index.tolist() == expected_index
    assert df_raw.columns.tolist() ==expected_columns
    assert df_hedge.shape == (8, 3)
    assert df_hedge.index.tolist() == expected_index
    assert df_hedge.columns.tolist() == expected_columns_hedge


"""
Test Mean Return
"""
def test_mean_return(df_raw, df_hedge):
    mean_return_raw = df_raw.loc['Mean return(% per day)'].astype(float)
    mean_return_hedge = df_hedge.loc['Mean return(% per day)'].astype(float)
    
    expected_mean_return_raw = [0.3, 0.18, 0.02]
    expected_mean_return_hedge = [0.29, 0.17, 0.01]
    
    assert np.allclose(mean_return_raw, expected_mean_return_raw, atol=0.02)
    assert np.allclose(mean_return_hedge, expected_mean_return_hedge, atol=0.02)

"""
Test standard deviation
"""
def test_std_dev(df_raw, df_hedge):
    std_dev_raw = df_raw.loc['Std.dev.(% per day)'].astype(float)
    std_dev_hedge = df_hedge.loc['Std.dev.(% per day)'].astype(float)
    
    expected_std_dev_raw = [0.56,0.61,0.52]
    expected_std_dev_hedge = [0.48,0.54,0.47]
    
    assert np.allclose(std_dev_raw, expected_std_dev_raw, atol=0.11)
    assert np.allclose(std_dev_hedge, expected_std_dev_hedge, atol=0.11)

"""
Test skewness
"""
def test_skewness(df_raw, df_hedge):
    skewness_raw = df_raw.loc['Skewness'].astype(float)
    skewness_hedge = df_hedge.loc['Skewness'].astype(float)
    
    expected_skewness_raw = [3.02,2.74,1.06]
    expected_skewness_hedge = [2.45,2.26,0.88]
    
    assert np.allclose(skewness_raw, expected_skewness_raw, atol=1)
    assert np.allclose(skewness_hedge, expected_skewness_hedge, atol=2)

"""
Test kurtosis
"""
def test_kurtosis(df_raw, df_hedge):
    kurtosis_raw = df_raw.loc['Kurtosis'].astype(float)
    kurtosis_hedge = df_hedge.loc['Kurtosis'].astype(float)
    
    expected_kurtosis_raw = [38.21,40.5,17.93]
    expected_kurtosis_hedge = [31.26,34.51,15.97]
    
    assert np.allclose(kurtosis_raw, expected_kurtosis_raw, rtol=5)
    assert np.allclose(kurtosis_hedge, expected_kurtosis_hedge, rtol=5)


"""
test Beta
"""
def test_beta(df_raw, df_hedge):
    beta_raw = df_raw.loc['Beta'].astype(float)
    beta_hedge = df_hedge.loc['Beta'].astype(float)
    
    expected_beta_raw = [0.11,0.11,0.09]
    expected_beta_hedge = [0,0,0]
    
    assert np.allclose(beta_raw, expected_beta_raw, atol=0.02)
    assert np.allclose(beta_hedge, expected_beta_hedge, atol=0)

"""
tes annualized sharpe ratio
"""
def test_annualized_sharpe_ratio(df_raw, df_hedge):
    sharpe_ratio_raw = df_raw.loc['Annualized Sharpe Ratio'].astype(float)
    sharpe_ratio_hedge = df_hedge.loc['Annualized Sharpe Ratio'].astype(float)
    
    expected_sharpe_ratio_raw = [8.44,4.5,0.56]
    expected_sharpe_ratio_hedge = [9.58,4.91,0.44]
    
    assert np.allclose(sharpe_ratio_raw, expected_sharpe_ratio_raw, rtol=0.2)
    assert np.allclose(sharpe_ratio_hedge, expected_sharpe_ratio_hedge, rtol=0.2)

"""
test worst day return
"""
def test_worst_day_return(df_raw, df_hedge):
    worst_day_return_raw = df_raw.loc['Worst day return(%)'].astype(float)
    worst_day_return_hedge = df_hedge.loc['Worst day return(%)'].astype(float)
    
    expected_worst_day_return_raw = [-3.88,-4.76,-3.93]
    expected_worst_day_return_hedge = [-2.26,-3.92,-3.12]
    
    assert np.allclose(worst_day_return_raw, expected_worst_day_return_raw, rtol=0.1)
    assert np.allclose(worst_day_return_hedge, expected_worst_day_return_hedge, rtol=0.4)

"""
worst 3-month return
"""
def test_worst_3_month_return(df_raw, df_hedge):
    worst_3_month_return_raw = df_raw.loc['Worst 3-month return(%)'].astype(float)
    worst_3_month_return_hedge = df_hedge.loc['Worst 3-month return(%)'].astype(float)
    
    expected_worst_3_month_return_raw = [2.56,-2.13,-9.28]
    expected_worst_3_month_return_hedge = [2.27,-1.28,-7.97]
    
    assert np.allclose(worst_3_month_return_raw, expected_worst_3_month_return_raw, rtol=0.4)
    assert np.allclose(worst_3_month_return_hedge, expected_worst_3_month_return_hedge, rtol=0.7)



if __name__ == '__main__':
    df_raw = calc_reversal_strategy.load_Table_1A()
    df_hedge = calc_reversal_strategy.load_Table_1B()

    # test format
    test_format(df_raw, df_hedge)

    # test mean return
    test_mean_return(df_raw, df_hedge)

    # test standard deviation
    test_std_dev(df_raw, df_hedge)

    # test beta
    test_beta(df_raw, df_hedge)

    # test annualized sharpe ratio
    test_annualized_sharpe_ratio(df_raw, df_hedge)

    # test worst day return
    test_worst_day_return(df_raw, df_hedge)

    # test worst 3-month return
    test_worst_3_month_return(df_raw, df_hedge)

    # test skewness
    test_skewness(df_raw, df_hedge)

    # test kurtosis
    test_kurtosis(df_raw, df_hedge)
