# Packages
import quandl
import numpy as np
import tensorflow
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScalar
quandl.ApiConfig.api_key = '<Gv9VqzUx_24QFyuG267H>'
# Imports data from quandl into an array

gold = quandl.get("LBMA/GOLD")
gold_data = np.array(gold)
silver = quandl.get("LBMA/SILVER")
silver_data = np.array(silver)
gold_usd_am = gold_data[:,0]

print(gold_data[:,0])
# Standardising the dataset
scaler = MinMaxScalar(feature_range=(0,1))
scaled_gold = scaler.fit_transform(gold_data.reshape(-1,1))
scaled_silver = scaler.fit_transform(silver_data.reshape(-1,1))
