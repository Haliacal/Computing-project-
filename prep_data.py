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

# Variables
count = 0
mid_prices = []
EMA = 0.0

test_data = quandl.get("LBMA/GOLD")
series = quandl.get("LBMA/GOLD")


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
unscaled_train_data = mid_prices[:13000]
unscaled_test_data = mid_prices[13000:]

# Normalising Data
scaler = MinMaxScaler()
train_data = np.array(unscaled_train_data).reshape(-1, 1)
test_data = np.array(unscaled_test_data).reshape(-1, 1)

# Smoothing
smoothing_window_size = 3500
for di in range(0, 10500, smoothing_window_size):
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
multiplier = 5

for ti in range(10500):
    EMA = multiplier * train_data[ti] + (1 - multiplier) * EMA
    train_data[ti] = EMA

# Combining data back together
all_mid_data = np.concatenate([train_data, test_data], axis=0)

plt.plot(range(gold_data.shape[0]),all_mid_data,color='blue',label='True')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.show()


'''
# Save contents
a = (np.array(all_mid_data)).reshape((len(all_mid_data),1))
file = open("mid_data.txt","w")
for row in a: np.savetxt(file, row)
file.close()
'''