# Packages
import quandl
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

#import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from numpy import *

quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'

plt.style.use('ggplot')

# Variables
count = 0
mid_prices = []
EMA = 0.0
gamma = 0.1
test_data = quandl.get("LBMA/GOLD")
series = quandl.get("LBMA/GOLD")


# Gamma = smoothing/(1+days)
''' Data Collection'''

# Import table data
gold_data = quandl.get("LBMA/GOLD", returns='numpy')

# Getting Open and close data
gold_usd_open = gold_data['USD (AM)']
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
''' Preparing Data'''
# Splitting training and testing data set
unscaled_train_data = mid_prices[:9000]
unscaled_test_data = mid_prices[9000:]

# Normalising Data
scaler = MinMaxScaler()
train_data = np.array(unscaled_train_data).reshape(-1, 1)
test_data = np.array(unscaled_test_data).reshape(-1, 1)

# Smoothing
smoothing_window_size = 2000
for di in range(0, 8000, smoothing_window_size):
	scaler.fit(train_data[di:di + smoothing_window_size, :])
	train_data[di:di + smoothing_window_size, :] = scaler.transform(
	    train_data[di:di + smoothing_window_size, :])

# Smoothing end of graph
scaler.fit(train_data[di + smoothing_window_size:, :])
train_data[di + smoothing_window_size:, :] = scaler.transform(
    train_data[di + smoothing_window_size:, :])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)
# The test data doesn't need to have windows as it is a much smaller dataset compared to the training data

for ti in range(8000):
	EMA = gamma * train_data[ti] + (1 - gamma) * EMA
	train_data[ti] = EMA

# Combining data back together
all_mid_data = np.concatenate([train_data, test_data], axis=0)

''' Averaging '''
# standard average

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    if pred_idx >= N:
    		date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
  
    else:
    	date = dateindex[pred_idx]

    std_avg_predictions.append(np.nan_to_num(np.mean(train_data[pred_idx-window_size:pred_idx])))
    mse_errors.append(np.nan_to_num((std_avg_predictions[-1]-train_data[pred_idx])**2))
    std_avg_x.append(date)


print('\nMSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
dateaxis = series.reset_index()['Date']

'''
plt.plot(dateaxis,all_mid_data,color='blue',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='red',label='Prediction')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.show()
'''
# https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp

# https://www.investopedia.com/terms/e/ema.asp

# https://www.datacamp.com/community/tutorials/lstm-python-stock-market
