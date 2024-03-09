import pandas as pd
import numpy as np
import pandas_datareader
import statsmodels.api as sm
import config
from pathlib import Path
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(config.DATA_DIR)
START_DATE = config.START_DATE
END_DATE = config.END_DATE
OUTPUT_DIR = Path(config.OUTPUT_DIR)

import calc_reversal_strategy
import load_vix

import numpy as np
import numpy as np
def performance_summary(return_data, annualization = 252):
    """ 
        Returns the Performance Stats for given set of returns
        Inputs: 
            return_data - DataFrame with Date index and Monthly Returns for different assets/strategies.
        Output:
            summary_stats - DataFrame with annualized mean return, vol, sharpe ratio. Skewness, Excess Kurtosis, Var (0.05) and
                            CVaR (0.05) and drawdown based on monthly returns. 
    """
    summary_stats = return_data.mean().to_frame('Annualized Mean Return(%)').apply(lambda x: x*annualization)
    summary_stats['Annualzied Volatility(%)'] = return_data.std().apply(lambda x: x*np.sqrt(annualization))
    summary_stats['Annualized Sharpe Ratio'] = summary_stats['Annualized Mean Return(%)']/summary_stats['Annualzied Volatility(%)']

    summary_stats['Skewness'] = return_data.skew()
    summary_stats['Kurtosis'] = return_data.kurtosis() + 3
    summary_stats['VaR (0.05)(%)'] = return_data.quantile(.05, axis = 0)
    summary_stats['CVaR (0.05)(%)'] = return_data[return_data <= return_data.quantile(.05, axis = 0)].mean()
    
    wealth_index = (1+return_data/100).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks

    summary_stats['Max Drawdown(%)'] = drawdowns.min()
    summary_stats = summary_stats.applymap('{:.2f}'.format)
    summary_stats['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
    summary_stats['Bottom'] = drawdowns.idxmin()
    
    recovery_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][:drawdowns[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin():]]).T
        recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
    summary_stats['Recovery Date'] = recovery_date

    summary_stats["Duration (days)"] = [
        (i - j).days if i != "-" else "-"
        for i, j in zip(summary_stats["Recovery Date"], summary_stats["Peak"])
    ]

    summary_stats["Peak"] = summary_stats["Peak"].dt.strftime('%Y-%m-%d')
    summary_stats["Bottom"] = summary_stats["Bottom"].dt.strftime('%Y-%m-%d')
    summary_stats["Recovery Date"] = summary_stats["Recovery Date"].dt.strftime('%Y-%m-%d')
    
    
    return summary_stats.T
    
    
    return summary_stats.T




if __name__ == "__main__":

    # Generate Additional Analysis Table
    ret_raw = calc_reversal_strategy.load_reversal_return(data_dir=DATA_DIR)
    index = pd.read_parquet(DATA_DIR / "pulled" / "CRSP_DSIX.parquet")
    index = index.set_index('caldt')['vwretx']*100
    strategies = pd.concat([ret_raw, index], axis=1)
    strategies.columns = ['Transact. prices','Quote-midpoints','Industry portfolio', 'CRSP Value Weighted Index']
    performance_matrix = performance_summary(strategies, 252)
    performance_matrix.to_parquet(DATA_DIR / "derived" / "Additional_Table.parquet")

    vix = load_vix.load_vix(data_dir=DATA_DIR)
    reversal_strategy = ret_raw.join(vix, how='left')
    reversal_strategy['Transact. prices'] = reversal_strategy['Transact. prices'].rolling(window=62).mean()

    # Plot reversal strategy and VIX
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(reversal_strategy.index, reversal_strategy['Transact. prices'], color='brown')
    ax1.set_ylabel('Average return per day')
    plt.legend(['Average return per day'])

    ax2 = ax1.twinx()
    ax2.plot(reversal_strategy.index, reversal_strategy['VIXCLS'])
    ax2.set_ylabel('VIX')
    plt.legend(['VIX'])

    ax1.set_xlabel('Date')

    plt.savefig(OUTPUT_DIR / 'reversal_strategy_vix.png');