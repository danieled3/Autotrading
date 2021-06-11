# Download 2 years of historical data of symbols in the configuration file and save output with a given name

# Import libraries
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import time
import json
import csv
import my_utils

# Set final table name
final_table_name = "df_test"

# Load alpha_vantage API key from config file
with open("config/config.json") as json_file:
    config_dict = json.load(json_file)
    apiKey = config_dict["alpha_vantage_password"]

# Load list of symbols to consider
symbols = np.array([])
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

# Download last 2 years data for each symbol
api_request_counter = 1

for symbol in [symbols[1]]:
    print(symbol)
    df_symbol = pd.DataFrame(columns=['time', symbol + '_close', symbol + '_volume'])

    for year in years:
        for month in months:
            print('Computation of year -' + str(year) + ' and month -' + str(month) + '...')
            slice = 'year' + str(year) + 'month' + str(month)
            df = my_utils.get_trading_data(ts, symbol, slice)
            df_symbol = df_symbol.append(df)  # append current df to full df_symbol

            if api_request_counter % 5 == 0:
                print('I am waiting for the API to reload...')
                time.sleep(59)  # wait for API to reload

            api_request_counter += 1

    df_tot = pd.merge(df_tot, df_symbol, on='time', how='left')

df_tot.to_csv('raw_data/' + final_table_name + '.csv')
