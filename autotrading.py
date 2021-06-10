# Select one symbol, download new data about the needed features with the right offset, make prediction and
# simulate investment according to deterministic rules

# Import libraries
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import pickle
import tensorflow as tf
import json
import utils

# Select symbol
symbol = 'TSLA_close'

# Load model info
with open('models/' + symbol + '_info.p', 'rb') as fp:
    model_info = pickle.load(fp)
days_to_predict = model_info['predicted_days']
window_size=model_info['window_size']
max_features_number = model_info['max_features_number']

#Load model
model = tf.keras.models.load_model('models/' + symbol + '_4pred.h5')

# Load alpha_vantage API key from config file
with open("config/config.json") as json_file:
    config_dict = json.load(json_file)
    apiKey = config_dict["alpha_vantage_password"]

# Initialize alpha_vantage object
ts = TimeSeries(key=apiKey, output_format='csv')

# Define time variables
now = datetime.now()  # current date and time
all_days = [(now - timedelta(i)).strftime("%Y-%m-%d") for i in range(1, 850) if
            (now - timedelta(i)).weekday() < 5]  # array of all days starting from yesterday (offset = 0)

# Load best features and best offset list
with open('features_selector_data/features_selector.p', 'rb') as fp:
    selected_features = pickle.load(fp)

features = selected_features[symbol]['features']
offsets = selected_features[symbol]['offset']

if max_features_number < len(features):  #consider only the features the model was trained with
    features = features[:max_features_number]
    offsets = offsets[:max_features_number]

x= [None] * len(features)

# Load new data of each features
for (i,feature) in enumerate(features):
    offset = offsets[i]
    days_needed = all_days[-offset:-offset-3]
    months_slice = list(set([ for days in days_needed]))
    year_slice = list(set([ for days in days_needed]))



    df = utils.get_trading_data(ts, symbol, slice)







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

