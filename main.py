# Packages
import quandl
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'

# Imports data from quandl into an array
gold_data = quandl.get("LBMA/GOLD")
gold_data = np.array(gold_data)

print(gold_data[:,0])

gold_usd_am = gold_data[:,0]

# Standardising the dataset
scaler = MinMaxScaler(feature_range=(0,1))
scaled_gold_usd_am = scaler.fit_transform(gold_usd_am.reshape(-1,1))


print(scaled_gold_usd_am)