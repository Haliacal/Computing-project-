# Packages
import quandl
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import keras

from matplotlib.pyplot import *
from sklearn.preprocessing import MinMaxScaler
from numpy import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'

plt.style.use('ggplot')

# Variables
count = 0
mid_prices = []
EMA = 0.0
gamma = 0.1
test_data = quandl.get("LBMA/GOLD")
series = quandl.get("LBMA/GOLD")
dateaxis = series.reset_index()['Date']

# Gamma = smoothing/(1+days)
''' Data Collection'''

# Import table data
gold_data = quandl.get("LBMA/GOLD", returns='numpy')

# Getting Open and close data
gold_usd_open = np.nan_to_num(gold_data['USD (AM)'])
gold_usd_close = np.nan_to_num(gold_data['USD (PM)'], nan=0.0)
dateindex = gold_data['Date']

# Getting mid values the open and close data
while (count < len(gold_usd_open)):
    temp = (gold_usd_open[count] + gold_usd_close[count]) / 2.0

    if (gold_usd_open[count] == 0 or gold_usd_close[count] == 0):
        mid_prices.append(temp * 2.0)

    else:
        mid_prices.append(temp)

    count = count + 1

count = 0

while(count < len(mid_prices)-1):
    if (mid_prices[count]):
        temp = (mid_prices[count-1] + mid_prices[count+1])/2.0
        mid_prices[count] = temp

    count = count + 1

''' Preparing Data'''
# Splitting training and testing data set
unscaled_train_data = mid_prices[:11000]
unscaled_test_data = mid_prices[11000:]

# Normalising Data
scaler = MinMaxScaler()
train_data = np.array(unscaled_train_data).reshape(-1, 1)
test_data = np.array(unscaled_test_data).reshape(-1, 1)

# Smoothing
smoothing_window_size = 2500
for di in range(0, 10000, smoothing_window_size):
    scaler.fit(train_data[di:di + smoothing_window_size, :])
    train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])


# Smoothing end of graph
scaler.fit(train_data[di + smoothing_window_size:, :])
train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

# Reshape both train and test data
train_data = np.array(train_data).reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)
# The test data doesn't need to have windows as it is a much smaller dataset compared to the training data

for ti in range(11000):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA


regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (train_data.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
date = range(0,train_data.size)
regressor.fit(train_data, date, epochs = 100, batch_size = 32)

print("done")