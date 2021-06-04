# Clear raw data and create correlation table

# Import libraries
import pandas as pd
import numpy as np
import scipy.stats

# Set parameters
last_day_to_consider = '2021-04-19'
#Load raw data
raw_data = pd.read_csv('raw_data/raw_data.csv').iloc[:,2:]

#Clear data
full_data = raw_data.dropna(axis=1,how='all') # drop completly empty columns
full_data = full_data[full_data.time<last_day_to_consider] # drop most recent rows (they may be uncompleted)
full_data = full_data.sort_values(by='time', axis=0, ascending=False) #sort values by time
full_data = full_data.drop(axis=1, labels='time') #delete time column
full_data = full_data.dropna(axis=0,how='all') #delete completly empty rows (bank holiday)
full_data = full_data.fillna(method='ffill', axis=0) #fill empty spot with the value of the previous day

def corr_offset(col1,col2,offset=0):
  if (offset == 0) :
    corr = scipy.stats.pearsonr(col1, col2)[0]
  else:
    corr = scipy.stats.pearsonr(col1[:-offset], col2[offset:])[0]
  return corr


columns_name = full_data.columns
corr_table = pd.DataFrame(columns=['Symbol_1', 'Symbol_2', 'Offset', 'Correlation'])
c1_i = 0
c2_i = 0

for column1_name in columns_name:
    column1 = full_data[column1_name]
    column1_length = sum(~np.isnan(column1))  # length without null values
    c1_i = c1_i + 1

    for column2_name in columns_name:
        column2 = full_data[column2_name]
        column2_length = sum(~np.isnan(column2))  # length without null values
        c2_i = c2_i + 1
        min_range = min(column1_length, column2_length)

        resized_column1 = column1[0:min_range]
        resized_column2 = column2[0:min_range]

        print('Index = ' + str(c1_i) + ' - Column1 = ' + column1_name)
        print('Index = ' + str(c2_i) + ' - Column2 = ' + column2_name)
        print('')

        for offset in range(min_range - 2):
            correlation = corr_offset(resized_column1, resized_column2, offset)
            new_row = {'Symbol_1': [column1_name], 'Symbol_2': [column2_name], 'Offset': [offset],
                       'Correlation': [abs(correlation)]}
            new_row_df = pd.DataFrame(new_row)
            corr_table = corr_table.append(new_row_df, ignore_index=True)
