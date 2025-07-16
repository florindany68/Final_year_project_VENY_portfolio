import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
ticker = 'WMT'
start_date = '2010-12-31'
end_date = '2024-12-31'

data = yf.download(ticker, start=start_date, end=end_date)

df_stock = data[['Close']].copy()

"""plt.figure(figsize=(12, 8))
plt.title(f"Histogram of {ticker}")
plt.plot(df_stock['Close'])
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()"""

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_stock)

def create_sequence(data, sequence_length):
    X = []
    Y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        Y.append(data[i + sequence_length])
    return np.array(X), np.array(Y)

sequence_length = 252


train_data_partition = int(len(scaled_data)*0.8)
train_data = scaled_data[:train_data_partition]
test_data = scaled_data[train_data_partition-sequence_length:]

train_end_date = df_stock.index[train_data_partition - 1]
test_start_date = df_stock.index[train_data_partition]

print(f"Training data ends on: {train_end_date}")
print(f"Testing data starts on: {test_start_date}")
X_train, y_train = create_sequence(train_data, sequence_length)
X_test, y_test = create_sequence(test_data, sequence_length)



# Reshape input for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=65, return_sequences=False))
model.add(Dense(units=20, activation='linear'))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)
model.save(f'LSTM_model_final.keras')
joblib.dump(scaler, 'LSTM_model_final_scaler.pkl')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()



predictions = model.predict(X_test)

# Inverse transform to get actual prices
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


rmse = np.sqrt(np.mean(((predictions - y_test_actual) ** 2)))
mae = np.mean((np.abs(predictions - y_test_actual)))
mape = np.mean(np.abs((predictions - y_test_actual) / np.maximum(np.abs(y_test_actual), 1e-8))) * 100
print(f"MAPE: {mape}")
print(f"MAE: {mae}")
print(f'RMSE: {rmse}')

train_predictions = model.predict(X_train)
train_predictions1 = scaler.inverse_transform(train_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
rmse_train = np.sqrt(np.mean(((train_predictions1 - y_train_actual) ** 2)))
mae_train = np.mean((np.abs(train_predictions1 - y_train_actual)))
print(f"MAE for training: {mae_train}")
print(f'RMSE for training: {rmse_train}')



# Predict first test day
first_sequence = test_data[:sequence_length]
first_sequence = first_sequence.reshape(1, sequence_length, 1)
first_test_day_prediction = model.predict(first_sequence)
first_test_day_prediction = scaler.inverse_transform(first_test_day_prediction)[0][0]

# Predict last test day
last_sequence = test_data[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, 1)
last_test_day_prediction = model.predict(last_sequence)
last_test_day_prediction = scaler.inverse_transform(last_test_day_prediction)[0][0]

# Calculate predicted return
predicted_return = ((last_test_day_prediction - first_test_day_prediction) / first_test_day_prediction) * 100

# Get actual prices for comparison
first_test_day_actual = df_stock.loc[test_start_date]['Close']
last_test_day_actual = df_stock.iloc[-1]['Close']
actual_return = ((last_test_day_actual - first_test_day_actual) / first_test_day_actual) * 100

print(f"Predicted first day of test price: ${first_test_day_prediction}")
print(f"Predicted last day of test price: ${last_test_day_prediction}")
print(f"Predicted return: {predicted_return}")
print(f"Actual first day price: ${first_test_day_actual}")
print(f"Actual last day price: ${last_test_day_actual}")
print(f"Actual return: {actual_return}")