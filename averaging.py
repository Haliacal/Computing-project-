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

all_mid_prices = np.loadtxt('mid_data.txt', dtype=float)
train_data = all_mid_prices[:13000]
test_data = all_mid_prices[13000:]

''' Standard Averaging '''

window_size = 300 # How many days into the past will be use to make interpolation prediction
N = train_data.size# Total data size

# Array variables
std_avg_predictions = []
std_avg_x = []
mse_total = []

for r in range(window_size,N):

    # Finding mean for next value
    mean = np.mean(train_data[r-window_size:r]) # finds mean through numpy of the last r days
    std_avg_predictions.append(np.nan_to_num(mean))

    # Calculating error
    mse = np.nan_to_num((std_avg_predictions[-1]-train_data[r])**2) # squares difference
    mse_total.append(mse)

    # Appending date
    std_avg_x.append(dateindex[r])

''' Exponential Moving average '''

date = []
N = train_data.size # Total data size

ema_avg_predictions = []
ema_avg_x = []
mse_errors = []

ema = 0.0
ema_avg_predictions.append(ema)

weight = 0.1

# Exponential moving average algorithm
for i in range(1,N):

    # Ema formula and appending values
    ema = ema*weight + (1-weight)*train_data[i-1]
    ema_avg_predictions.append(ema)

    # calculating error and appending date
    mse_errors.append((ema_avg_predictions[-1]-train_data[i])**2)
    ema_avg_x.append(date)

print('MSE error: %.5f'%(0.5*np.mean(mse_errors)))

plt.plot(range(gold_data.shape[0]),all_mid_prices,color='blue',label='True')
plt.plot(range(0,N),ema_avg_predictions,color='orange', label='EMA')
plt.plot(range(window_size,N),std_avg_predictions,color='green',label='SMA')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend()
plt.show()


# https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp

# https://www.investopedia.com/terms/e/ema.asp

# https://www.datacamp.com/community/tutorials/lstm-python-stock-market
