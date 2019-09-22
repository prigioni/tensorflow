#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019-9-22 21:29 
# @Author : lauqasim
# @File : stock_predict_keras.py 
# @Software: PyCharm
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas_datareader.data as web
import datetime
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
style.use('ggplot')

# get 2014-2018 data to train our model
start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2018, 12, 30)
df = web.DataReader("TSLA", 'yahoo', start, end)

# get 2019 data to test our model on
start = datetime.datetime(2019, 1, 1)
end = datetime.date.today()
test_df = web.DataReader("TSLA", 'yahoo', start, end)
# sort by date
df = df.sort_values('Date')
test_df = test_df.sort_values('Date')
# fix the date
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)
test_df.reset_index(inplace=True)
test_df.set_index("Date", inplace=True)

# change the dates into ints for training
dates_df = df.copy()
dates_df = dates_df.reset_index()

# Store the original dates for plotting the predicitons
org_dates = dates_df['Date']
# convert to ints
dates_df['Date'] = dates_df['Date'].map(mdates.date2num)

# Create train set of adj close prices data:
train_data = df.loc[:, 'Adj Close'].as_matrix()
# print(train_data)
# Apply normalization before feeding to LSTM using sklearn:


scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)

scaler.fit(train_data)
train_data = scaler.transform(train_data)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Create the data to train our model on:
time_steps = 36
X_train, y_train = create_dataset(train_data, time_steps)

# reshape it [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 36, 1))


# Visualizing our data with prints:
print(str(scaler.inverse_transform(X_train[0])))
print('y_train: ' + str(scaler.inverse_transform(y_train[0].reshape(-1, 1))) + '\n')
print('y_train.shape', y_train.shape)
# Build the model
model = keras.Sequential()

model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the model to the Training set
history = model.fit(X_train, y_train, epochs=2, batch_size=10, validation_split=.30)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Get the stock prices for 2019 to have our model make the predictions
test_data = test_df['Adj Close'].values
test_data = test_data.reshape(-1, 1)
test_data = scaler.transform(test_data)

# Create the data to test our model on:
time_steps = 36
X_test, y_test = create_dataset(test_data, time_steps)

# store the original vals for plotting the predictions
y_test = y_test.reshape(-1, 1)
org_y = scaler.inverse_transform(y_test)

# reshape it [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], 36, 1))

# Predict the prices with the model
predicted_y = model.predict(X_test)
predicted_y = scaler.inverse_transform(predicted_y)

# plot the results
plt.plot(org_y, color='red', label='Real Tesla Stock Price')
plt.plot(predicted_y, color='blue', label='Predicted Tesla Stock Price')
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.show()