# Clear raw data and create correlation table

# Import libraries
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta

# Set parameters
last_day_to_consider = '2021-04-19'
symbols_per_file = 4


# Define function to compute Pearson correlation between a Pandas Series col1  and another Pandas series col2
# shifted by offset (offset must be >= 0)
def corr_offset(col1, col2, offset=0):
    if offset > 0:
        corr = np.corrcoef(col1[:-offset], col2[offset:])[0, 1]
    else:  # it means offset =0
        corr = np.corrcoef(col1, col2)[0, 1]
    return corr


# Load DataFrames of raw data
files = [f for f in listdir('raw_data') if isfile(join('raw_data', f))]
now = datetime.now()
all_days = [(now - timedelta(i)).strftime("%Y-%m-%d") for i in range(0, 1000) if
            (now - timedelta(i)).weekday() < 5]  # array of all days

full_data = pd.DataFrame(all_days, columns=['time'])

for f in files:  # load all csv and merge them into a unique DataFrame
    current_df = pd.read_csv(join('raw_data', f))
    full_data = pd.merge(left=full_data, right=current_df, how='outer', on='time',
                         suffixes=('', '_delete'))  # mark double columns

# Clear data and column names
new_columns_list = []

for column_name in full_data.columns:  # create list of clear column names
    if column_name.endswith('_x') or column_name.endswith('_y'):
        column_name = column_name[0:-2]
    new_columns_list = new_columns_list + [column_name]

full_data.columns = new_columns_list  # rename columns
full_data = full_data.drop(labels='Unnamed: 0', axis=1)  # drop row number

for column_name in list(set(full_data.columns)):  # delete double columns
    if column_name.endswith('_delete'):
        full_data = full_data.drop(labels=column_name, axis=1)

# Manage NaN
full_data = full_data.dropna(axis=1, how='all')  # drop completely empty columns
full_data = full_data[full_data.time < last_day_to_consider]  # drop most recent rows (they may be uncompleted)
full_data = full_data.sort_values(by='time', axis=0, ascending=False)  # sort values by time
full_data = full_data.drop(axis=1, labels='time')  # delete time column
full_data = full_data.dropna(axis=0, how='all')  # delete completely empty rows (bank holiday)
full_data = full_data.fillna(method='ffill', axis=0)  # fill empty spot with the value of the previous day

# Save full data
full_data.to_csv('clear_data/full_data.csv')

# Compute correlation table
column_names = full_data.columns
close_names = [col for col in column_names if col.endswith('_close')]
symbol1_array = np.array([])
symbol2_array = np.array([])
offset_array = np.array([])
correlation_array = np.array([])
symbol_list = np.array([])
c1_i = 0  # index of first symbol
c2_i = 0  # index of second symbol

for column1_name in close_names:  # iteration only on close columns (we do not want to predict volume)
    c1_i = c1_i + 1
    c2_i = 0
    column1 = np.array(full_data[column1_name])
    column1_length = sum(~np.isnan(column1))  # length without null values
    symbol_list = np.append(symbol_list, column1_name.split('_')[0])

    for column2_name in column_names:
        c2_i = c2_i + 1  # update index
        column2 = np.array(full_data[column2_name])
        column2_length = sum(~np.isnan(column2))  # length without null values
        min_range = min(column1_length, column2_length)
        max_offset = min_range - 2

        resized_column1 = column1[0:min_range]
        resized_column2 = column2[0:min_range]

        symbol1_repeated = np.repeat(column1_name, max_offset)
        symbol2_repeated = np.repeat(column2_name, max_offset)
        offset_range = np.arange(max_offset)

        symbol1_array = np.concatenate((symbol1_array, symbol1_repeated))  # generate column values by repetition
        symbol2_array = np.concatenate((symbol2_array, symbol2_repeated))  # generate column values by repetition
        offset_array = np.concatenate((offset_array, offset_range)) # the column value is a set of consecutive numbers

        print('Index = ' + str(c1_i) + ' - Column1 = ' + column1_name)
        print('Index = ' + str(c2_i) + ' - Column2 = ' + column2_name)
        print('')

        for offset in range(min_range - 2): # we do not consider correlation between 2-length array
            correlation = corr_offset(resized_column1, resized_column2, offset)
            correlation_array = np.append(correlation_array, correlation)

    if len(symbol_list) == symbols_per_file: # Save a table every "symbol_per_file" rows to delete columns values and speed up computation
        corr_table = pd.DataFrame({
            'Symbol_1': symbol1_array,
            'Symbol_2': symbol2_array,
            'Offset': offset_array,
            'Correlation': correlation_array
        })

        output_name = '_'.join(symbol_list)
        corr_table.to_csv('clean_data/corr_' + output_name + '.csv') # Use names of symbols as file name

        symbol1_array = np.array([])
        symbol2_array = np.array([])
        offset_array = np.array([])
        correlation_array = np.array([])
        symbol_list = np.array([])
