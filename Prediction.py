# Packages
import quandl
import pandas as pd
import datetime as dt
import numpy as np

#import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from numpy import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties

quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'
plt.style.use('ggplot')

# Variables
count = 0
mid_prices = []

# Import table data
gold_data = quandl.get("LBMA/GOLD",returns='numpy')
test_data = quandl.get("LBMA/GOLD")
df = pd.DataFrame(test_data)

# Yearly Data

# Getting Open and close data
gold_usd_open = gold_data['USD (AM)']
gold_usd_close = np.nan_to_num(gold_data['USD (PM)'], nan=0.0)

# Getting mid values the open and close data
while (count < len(gold_usd_open)):
    temp = (gold_usd_open[count] + gold_usd_close[count])/2.0

    if(gold_usd_open[count] == 0 or gold_usd_close[count] == 0 ):
        mid_prices.append(temp*2.0)

    else:
        mid_prices.append(temp)
    count = count + 1

# Splitting training and testing data set
unscaled_train_data = mid_prices[:11000]
unscaled_test_data = mid_prices[11000:]

# Normalising Data
scaler = MinMaxScaler()
train_data = np.reshape(unscaled_test_data, (-1,1))
test_data = np.reshape(unscaled_test_data, (-1,1))

# unscaled_train_data.reshape(-1,1)

# Smoothing
smoothing_window_size = 2500
for di in range(0,10000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
    end = di

# Normalising the end bit of the data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

print("done")