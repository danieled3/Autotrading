# Compute correlation table where we can see, for each symbol, the most correlated symbols and the best time offset

# Import libraries
import pandas as pd
import numpy as np
import os

# Set minimum offset to consider (max prediction time)
min_offset = 20

# Load full data
full_data = pd.read_csv('clean_data/full_data.csv')

# Extract features list and symbols list
features_names = full_data.columns  # list of all features
symbol_names = [col for col in features_names if col.endswith('_close')]  # list of symbols

# Create best correlation and best offset tables
best_correlation_df = pd.DataFrame(index=features_names, columns=symbol_names)
best_correlation_df = best_correlation_df.fillna(0.)
best_offset_df = pd.DataFrame(index=features_names, columns=symbol_names)
best_offset_df = best_offset_df.fillna(0)

# Find file to load
symbols_corr_tables = [f for f in os.listdir('clean_data') if f.startswith('corr_')]
number_of_tables = len(symbols_corr_tables)
current_table_index = 1

# Computation of best correlation and offset
for table in symbols_corr_tables:
    print('Computing table ' + str(current_table_index) + ' of ' + str(number_of_tables) + '...')
    current_symbols_corr = pd.read_csv('clean_data/' + table)
    current_symbols_corr = current_symbols_corr[
        np.logical_and(current_symbols_corr.Offset >= min_offset, current_symbols_corr.Offset <= 240)]
    current_symbols_corr['Correlation'] = current_symbols_corr['Correlation'].fillna(0)

    for row in current_symbols_corr.itertuples():
        symbol_1 = row[2]
        symbol_2 = row[3]
        offset = row[4]
        correlation = row[5]
        if abs(correlation) > abs(best_correlation_df[symbol_1][symbol_2]):
            best_correlation_df[symbol_1][symbol_2] = correlation
            best_offset_df[symbol_1][symbol_2] = offset
    print('Table ' + str(current_table_index) + ' of ' + str(number_of_tables) + ' computed!')
    current_table_index += 1

best_correlation_df.to_csv('correlation_data/correlation.csv')
best_offset_df.to_csv('correlation_data/offset.csv')