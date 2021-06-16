# Select one symbol, download new data about the needed features with the right offset, make prediction and
# simulate investment according to deterministic rules

# Import libraries
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from dateutil import relativedelta
import time
import pickle
from tensorflow.keras.models import load_model
import json
import my_utils
import telegram
import matplotlib.pyplot as plt

# Select symbol
symbol = 'TSLA_close'

# Load model info
with open('models/' + symbol + '_info.p', 'rb') as fp:
    model_info = pickle.load(fp)
days_to_predict = model_info['predicted_days']
window_size = model_info['window_size']
max_features_number = model_info['max_features_number']

# Load model
model = load_model('models/' + symbol + '.h5')

# Load alpha_vantage API key, token for telegram bot and chat id from config file
with open("config/config.json") as json_file:
    config_dict = json.load(json_file)
    apiKey = config_dict["alpha_vantage_password"]
    chat_id = config_dict["chat_id"]
    token = config_dict["bot_token"]

# Start bot
bot = telegram.Bot(token)
text = 'Good evening! I am Trady, your personal trading assistant!'
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)

text = 'I am analysing worldwide transactions to provide you the best advice. Please wait for 15-20 minutes...'
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)
bot.sendAnimation(chat_id, animation=open('charts/loading_data.gif', 'rb'))

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
symbol_only_name = symbol.split('_')[0]

if max_features_number < len(features):  # consider only the features the model was trained with
    features = features[:max_features_number]
    offsets = offsets[:max_features_number]

x = [None] * len(features)
api_request_counter = 0

# Load new data of each features
for (i, feature) in enumerate(features):
    feature_only_name = feature.split('_')[0]
    offset = offsets[i]
    needed_days = all_days[offset:offset + window_size]  # select the days that we want to download
    print(needed_days)

    # since the data we want may be in the previous or in the following "slice" we download data also
    # for some days more
    safe_bound = 3
    needed_days_safe = all_days[offset - safe_bound:offset + window_size + safe_bound]
    months_slice = list(set([relativedelta.relativedelta(now, datetime.strptime(day, '%Y-%m-%d')).months + 1 for day in
                             needed_days_safe]))
    years_slice = list(set([relativedelta.relativedelta(now, datetime.strptime(day, '%Y-%m-%d')).years + 1 for day in
                            needed_days_safe]))

    df_feature = pd.DataFrame(columns=['time', feature_only_name + '_close', feature_only_name + '_volume'])

    # Download all needed data
    for year in years_slice:
        for month in months_slice:
            slice = 'year' + str(year) + 'month' + str(month)
            df = my_utils.get_trading_data(ts, feature_only_name, slice)  # API call
            api_request_counter += 1
            if api_request_counter % 5 == 0:
                time.sleep(60)  # wait for API to reload
            df_feature = pd.concat([df_feature, df])  # add data

    needed_values = np.array(df_feature[df_feature['time'].isin(needed_days)][feature])  # select only needed days
    if len(needed_values)==0:
        needed_values = np.array([max(0,np.mean(df_feature[feature].astype('float64')))]) #use mean in case of not found value
    needed_values = needed_values.astype('float64')
    while len(needed_values)<window_size: # fill empty values with the previous (there may be nans in bank holiday days)
        needed_values = np.append(needed_values,needed_values[-1])
    x[i] = needed_values  # add values of this features to x list of array

x = np.array(x)  # transform x into a numpy array
prediction = model.predict(x.T.reshape(1,window_size,len(features)))[0][0]

# Send another message
text = 'I am almost done! Please just give me another 5 minutes...'
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)
bot.sendAnimation(chat_id, animation=open('charts/loading_data_2.gif', 'rb'))


# Load asset
months_to_load = 8 # download last months of selected symbol
df_current_symbol = pd.DataFrame(columns=['time', symbol])

for month in range(months_to_load, 0, -1):
    slice = 'year1month' + str(month)
    if month % 4 == 0:  # start with sleep
        time.sleep(59)
    df = my_utils.get_trading_data(ts, symbol_only_name, slice)  # API call
    df_current_symbol = pd.concat([df_current_symbol, df[['time', symbol]]])

df_current_symbol = df_current_symbol.sort_values('time')

# Load asset info
portfolio = pd.read_csv('portfolio/portfolio.csv', index_col=False)

# Compute significant quantities
long_average_days = 90
short_average_days = 30
time_array = np.array(df_current_symbol['time']).astype('object')
series = np.array(df_current_symbol[symbol]).astype('float64')  # the first value is the oldest one
long_averages = my_utils.moving_average(series, long_average_days)
short_averages = my_utils.moving_average(series, short_average_days)
long_average_dby = long_averages[-2]  # long average of the day before yesterday
long_average_yesterday = long_averages[-1]  # long average of yesterday
short_average_dby = short_averages[-2]  # long average of the day before yesterday
short_average_yesterday = short_averages[-1]  # long average of yesterday
yesterday_symbol_value = series[-1]
liquid_value = portfolio[portfolio['Asset'] == 'Liquid']['Value'].iloc[0]
asset_value = portfolio[portfolio['Asset'] == symbol_only_name]['Value'].iloc[0]
new_liquid_value = liquid_value
new_asset_value = asset_value


# Send info about current portfolio
text = 'The current composition in dollars of your portfolio is the following:'
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)

text = 'Liquid = ' + str(liquid_value) + ' dollars'
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)

text = str(symbol_only_name) + ' = ' + str(asset_value * yesterday_symbol_value) + ' dollars'
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)

# Send image of current portfolio
objects = ('Liquid', symbol_only_name)
y_pos = np.arange(len(objects))
performance = [liquid_value, asset_value * yesterday_symbol_value]

plt.figure(0)
plt.barh(y_pos, performance, align='center', alpha=0.8)
plt.yticks(y_pos, objects)
plt.xlabel('Value (Dollars)')
plt.title('Your current PORTFOLIO')

plt.savefig('charts/portfolio.png')
bot.send_photo(chat_id, photo=open('charts/portfolio.png', 'rb'))

time.sleep(5)

# Send info about analyzed symbol
text = 'The index I am analyzing is ' + symbol_only_name
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)

text = 'The value of ' + symbol_only_name + ' in the last ' + str(months_to_load) + ' has been the following one:'
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)

# Send trend of analyzed symbol
plt.figure(1)
plt.plot(series, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4, label=symbol)
plt.plot(long_averages, marker='', color='red', linewidth=2, linestyle='dashed', label=symbol + ' long averages')
plt.plot(short_averages, marker='', color='green', linewidth=2, linestyle='dashed', label=symbol + ' short averages')
plt.xlabel('Time')
plt.ylabel('Value')
plt.scatter(len(series) + days_to_predict[-1], prediction, edgecolors='red', label='Prediction')
plt.legend()
plt.grid()
plt.show()
plt.savefig('charts/' + symbol_only_name + '_trend.png')
bot.send_photo(chat_id, photo=open('charts/' + symbol_only_name + '_trend.png', 'rb'))

# Send the prediction value
text = 'As you can see, I predict that the value of ' + symbol_only_name + ' in ' + str(days_to_predict[
    -1]) + ' days will be ' + str(prediction)
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)

time.sleep(5)

# Buy when the shorter-term MA cross above the longer-term MA and the prediction value is higher

if (long_average_dby > short_average_dby
        and long_average_yesterday < short_average_yesterday
        and yesterday_symbol_value < prediction
):
    if prediction > yesterday_symbol_value * 1.3:  # buy with 50% of liquid value
        value_to_invest = int(liquid_value * 0.5)
        text = 'Since the shorter-term MA crossed above the longer-term MA and my prediction is very high, I will buy ' \
               + str(value_to_invest) + ' euros of ' + symbol_only_name + ' stocks!'
    else:  # buy with 30% of liquid value
        value_to_invest = int(liquid_value * 0.3)
        text = 'Since the shorter-term MA crossed above the longer-term MA, I will buy ' + str(
            value_to_invest) + ' euros of ' + symbol_only_name + ' stocks!'
    new_asset_value = asset_value + value_to_invest / yesterday_symbol_value
    new_liquid_value = liquid_value - value_to_invest

elif (long_average_dby < short_average_dby
      and long_average_yesterday > short_average_yesterday
      and yesterday_symbol_value > prediction
):
    if (prediction < yesterday_symbol_value * 0.8):  # sell 100% of asset
        value_to_sell = asset_value
        text = 'Since the shorter-term MA crossed below the longer-term MA and my prediction is very low, I will sell ' \
               + value_to_sell + ' ' + symbol_only_name + ' stocks!'
    else:  # sell 75% of asset
        value_to_sell = int(asset_value * 0.75)
        text = 'Since the shorter-term MA crossed below the longer-term MA, I will sell ' + str(
            value_to_sell) + ' ' + symbol_only_name + ' stocks!'
    new_asset_value = asset_value - value_to_sell
    new_liquid_value = liquid_value + value_to_sell * yesterday_symbol_value
else:
    text = 'Since the shorter-term MA is greatly different from longer-term MA, I would prefer to wait...'

bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)

text = 'Have a good day! See you tomorrow!'
bot.send_message(chat_id, text, parse_mode='markdown', disable_web_page_preview=True)
