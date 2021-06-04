# Clear raw data and create correlation table

# Import libraries
import pandas as pd
import numpy as np
import scipy.stats

#Load raw data
df = pd.read_csv('raw_data/raw_data.csv')

new_columns_list = []
for column_name in df.columns:
    if column_name.endswith('_x') or column_name.endswith('_y'):
        column_name = column_name[0:-2]

    new_columns_list = new_columns_list + [column_name]

df.columns = new_columns_list  # rename columns
df = df.drop(labels='Unnamed: 0', axis=1)  # drop row number

# delete double columns
for column_name in list(set(df.columns)):
  if column_name.endswith('_delete'):
    df = df.drop(labels=column_name,axis=1)

df_final = df.dropna(axis=1,how='all')
df_final = df_final[df_final.time<'2021-04-19']
df_final = df_final.sort_values(by='time', axis=0, ascending=False)
df_final = df_final.drop(axis=1, labels='time')
df_final = df_final.dropna(axis=0,how='all')
df_final = df_final.fillna(method='ffill', axis=0)

def corr_offset(col1,col2,offset=0):
  if (offset == 0) :
    corr = scipy.stats.pearsonr(col1, col2)[0]
  else:
    corr = scipy.stats.pearsonr(col1[:-offset], col2[offset:])[0]
  return corr


columns_name = df_final.columns
corr_table = pd.DataFrame(columns=['Symbol_1', 'Symbol_2', 'Offset', 'Correlation'])
c1_i = 0
c2_i = 0

for column1_name in columns_name:
    column1 = df_final[column1_name]
    column1_length = sum(~np.isnan(column1))  # length without null values
    c1_i = c1_i + 1

    for column2_name in columns_name:
        column2 = df_final[column2_name]
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
