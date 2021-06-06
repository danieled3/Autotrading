# Find for each symbol the most correlated features to use in the model
# The correlation between each features is not computed and will be managed in the model building

# Import libraries
import pandas as pd
import numpy as np
import pickle

# Set minimum correlation needed for a feature to be considered
min_corr = 0.85

# Load correlation table and offset table
best_correlation_df = pd.read_csv('correlation_data/correlation.csv', index_col=0)
best_offset_df = pd.read_csv('correlation_data/offset.csv', index_col=0)
symbol_names = best_correlation_df.columns

# Create dictionary with best features, offsets and  importance for each symbol
# best features: features with a correlation grater than min_corr
# best_offset: corresponding offset of best features
# average_corr: average correlation of the best features
selected_features = {
    symbol: {
        'features': np.array([]),
        'offset': np.array([])
    }
    for symbol in symbol_names
}

for symbol in symbol_names:
    best_features = best_correlation_df[symbol][
        abs(best_correlation_df[symbol]) > min_corr]  # select the best features
    selected_features[symbol]['features'] = list(best_features.index)  # collect the best features name
    selected_features[symbol]['offset'] = list(
        best_offset_df[symbol][best_features.index])  # collect the best offset of the best features

with open('features_selector_data/features_selector.p', 'wb') as fp:
    pickle.dump(selected_features, fp)