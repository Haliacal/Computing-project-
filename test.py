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
gold_usd_open = np.nan_to_num(gold_data['USD (AM)'])
gold_usd_close = np.nan_to_num(gold_data['USD (PM)'], nan=0.0)
dateindex = gold_data['Date']

array_sum = np.sum(gold_usd_close)
array_has_nan = np.isnan(array_sum)
print("gold_usd_close data: ", array_has_nan)

array_sum = np.sum(gold_usd_open)
array_has_nan = np.isnan(array_sum)
print("gold_usd_open data: ", array_has_nan)

# Getting mid values the open and close data
while (count < len(gold_usd_open)):
	temp = (gold_usd_open[count] + gold_usd_close[count]) / 2.0

	if (gold_usd_open[count] == 0 or gold_usd_close[count] == 0):
		mid_prices.append(temp * 2.0)

	else:
		mid_prices.append(temp)

	count = count + 1

count = 0

while(count < len(mid_prices)):
    if (mid_prices[count]):
    	temp = (mid_prices[count-1] + mid_prices[count+1])/2.0
        mid_prices[count] = temp

    count = count + 1

array_sum = np.sum(mid_prices)
array_has_nan = np.isnan(array_sum)
print("Mid data: ", array_has_nan)