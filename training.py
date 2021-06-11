# Train a model for each symbol by using best features and best offset.
# For each symbol we want to predict the close value after 5,10,15,20 days

# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# Select symbol to predict
symbol = 'TSLA_close'
max_features_number = 60  # limit the max number of features to use, we can download a limited amount of
# data because of API boundary
train_split = 0.8
days_to_predict = [20]
window_size = 3
maximum_mae_admitted = 48
patience = 300  # patience of the Early Stopping Checkout

# Load best features and best offset list
with open('features_selector_data/features_selector.p', 'rb') as fp:
    selected_features = pickle.load(fp)

features = selected_features[symbol]['features']
offset = selected_features[symbol]['offset']

# Load data
full_data = pd.read_csv('clean_data/full_data.csv', index_col=False)

# I needed, select only the first max_features_number features that are the most correlated ones
if len(selected_features[symbol]['features']) > max_features_number:
    features = selected_features[symbol]['features'][:max_features_number]
    offsets = selected_features[symbol]['offset'][:max_features_number]
else:
    features = selected_features[symbol]['features']
    offsets = selected_features[symbol]['offset']

# Create matrices for x and y where y[time] and x[time,feature]
y = np.flip(np.array(full_data[symbol]))  # y[time]
x = np.array(full_data[features])
for i in range(x.shape[1]):  # x[time,feature]
    x[:, i] = np.flip(x[:, i])

# Apply offset to each features in x and reshape matrix
series_length = [(x.shape[0] - offsets[i] - sum(np.isnan(x[i])))
                 for i in range(len(features))
                 ]
x_length = min(series_length)
offset_y = y[-x_length:]
offset_x = np.empty((x_length, len(features)))

for i in range(len(features)):
    offset_x[:, i] = x[:, i][-offsets[i] - x_length: -offsets[i]]

max_predicted_period = max(days_to_predict)
window_x = [None] * (x_length - window_size)
window_y = [None] * (x_length - window_size)

for i in range(x_length - window_size):
    current_window_x = offset_x[i + 1:i + window_size + 1, :]
    current_window_y = offset_y[i + window_size - max_predicted_period + np.array(days_to_predict)]
    window_x[i] = current_window_x
    window_y[i] = current_window_y

window_x = np.array(window_x)
window_y = np.array(window_y)

# Training-Validation split: we use the oldest values for training and the newest values for validation
index_split = int(train_split * window_x.shape[0])
x_train = window_x[:index_split, :, :]
x_valid = window_x[index_split:, :, :]
y_train = window_y[:index_split, :]
y_valid = window_y[index_split:, :]

# Model training
optimizer = tf.keras.optimizers.Adam(lr=1e-5)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('models/' + symbol + '.h5', save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=300, restore_best_weights=True)

# since model performances are very linked to selection of initial values, the model is trained
# until it has a particular loss value
go_on = True

while go_on:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[window_size, len(features)]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(len(days_to_predict)),
        tf.keras.layers.Lambda(lambda x: x * max(y))  # depends on data
    ])

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        epochs=5000,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    if history.history['val_mae'][-patience] < maximum_mae_admitted:  # [-patience] is the minimum loss
        go_on = False


# Plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))  # get number of epochs

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.vlines(x=len(loss) - 300, ymin=0, ymax=600,  # add early stopping vertical line
           colors='green', ls=':', lw=2,
           label='Early stopping')
plt.title('Training and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.figure()

# Save model info
model_info = {
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'predicted_days': days_to_predict,
    'window_size': window_size,
    'max_features_number': max_features_number
}

with open('models/' + symbol + '_info.p', 'wb') as fp:
    pickle.dump(model_info, fp)
