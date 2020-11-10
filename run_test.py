# Packages
import quandl
import pandas as pd
import datetime as dt
import numpy as np
#import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from numpy import *

quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'

# Variables
count = 0
mid_prices = []

# Imports data from quandl into an array
gold_data = quandl.get("LBMA/GOLD",returns='numpy')


gold_usd_open = gold_data['USD (AM)']
gold_usd_close = np.nan_to_num(gold_data['USD (PM)'], nan=0.0)


print(len(gold_usd_open))
print(len(gold_usd_close))


while (count < len(gold_usd_open)):
    temp = (gold_usd_open[count] + gold_usd_close[count])/2.0

    if(gold_usd_open[count] == 0 or gold_usd_close[count] == 0 ):
        mid_prices.append(temp*2.0)

    else:
        mid_prices.append(temp)
    count = count + 1

print(mid_prices)

