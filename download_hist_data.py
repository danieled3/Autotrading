# This code download 2 years of historical data of symbols in the configuration file and save output with a given name

# Import libraries
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import time
import json
import csv

# Set final table name
final_table_name = "df_1"

# Load alpha_vantage API key from config file
with open("config/config.json") as json_file:
    config_dict = json.load(json_file)
    apiKey = config_dict["alpha_vantage_password"]

# Load list of indexes to consider
with open('config/indexes_to_consider.txt', 'r') as txt_file:
    reader = csv.reader(txt_file)
    for row in reader:
        symbols = np.append(symbols, row)
symbols = list(set(symbols))  # delete multiple symbols

# Initialize alpha_vantage object
ts = TimeSeries(key=apiKey, output_format='csv')

# Define time variables
years = list(range(2, 0, -1))
months = list(range(12, 0, -1))
now = datetime.now()  # current date and time
all_days = [(now - timedelta(i)).strftime("%Y-%m-%d") for i in range(0, 850) if
            (now - timedelta(i)).weekday() < 5]  # array of all days

# Initialize final DataFrame
df_tot = pd.DataFrame(all_days, columns=['time'])

counter = 1

# Download last 2 years data for each symbol
for symbol in symbols:
    print(symbol)
    df_symbol = pd.DataFrame(columns=['time', symbol + '_close', symbol + '_volume'])

    for year in years:
        for month in months:
            print('Computation of year -' + str(year) + ' and month -' + str(month) + '...')
            slice = 'year' + str(year) + 'month' + str(month)

            totalData = ts.get_intraday_extended(symbol=symbol, interval='60min', slice=slice)  # download the csv
            df = pd.DataFrame(list(totalData[0]))  # csv --> dataframe

            header_row = 0
            df.columns = df.iloc[header_row]  # set column header
            df = df.drop(header_row)
            df = df.reset_index()
            df['time'] = [t[0:10] for t in df.time]  # extract date from datetime
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')  # transform string into integer
            df = df[['time', 'close', 'volume']].groupby('time').agg(
                {'close': ['first'], 'volume': ['sum']}).reset_index()  # aggregate on day
            df.columns = ['time', symbol + '_close', symbol + '_volume']  # rename columns
            df_symbol = df_symbol.append(df)  # append current df to full df_symbol

            if counter % 5 == 0:
                print('I am waiting for the API to reload...')
                time.sleep(59)  # wait for API to reload

            counter += 1

    df_tot = pd.merge(df_tot, df_symbol, on='time', how='left')

df_tot.to_csv('raw_data/' + final_table_name + '.csv')
