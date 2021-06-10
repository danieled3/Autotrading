# Collection of useful functions

# Import libraries
import pandas as pd

"""Download close and volume data of a particular symbol in a particular period and return them as a DataFrame

Arguments:
    symbol: index to download
    ts: Timeseries object of alpha_vantage library to use 
    slice: month to download (format= yearYmonthM)
    
Return:
    The Dataframe with time, close and volume info on daily basis"""

def get_trading_data(ts, symbol, slice):
    total_data = ts.get_intraday_extended(symbol=symbol, interval='60min', slice=slice)  # download the csv
    df = pd.DataFrame(list(total_data[0]))  # csv --> dataframe

    df.columns = df.iloc[0]  # set column header
    df = df.drop(0)  # drop header row
    df = df.reset_index()
    df['time'] = [t[0:10] for t in df.time]  # extract date from datetime
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')  # transform string into integer
    df = df[['time', 'close', 'volume']].groupby('time').agg(
        {'close': ['first'], 'volume': ['sum']}).reset_index()  # aggregate on day
    df.columns = ['time', symbol + '_close', symbol + '_volume']  # rename columns
    return df
