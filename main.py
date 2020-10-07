# Packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import quandl
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'

# Separates the dates from the metal price
def import_data(time_or_price, metal_data):
    metal = []
    count = 0
    while(count != (len(metal_data))):
        metal.append(metal_data[count][time_or_price])
        count = count + 1
    np.array(metal)
    return metal;

# Variables
date = 0
price = 1

# Imports data from quandl into an array
gold_data = quandl.get("LBMA/GOLD")

gold_usd_am = import_data(price,quandl.get('LBMA/GOLD', column_index='1', returns='numpy'))
metal_date = import_data(date,quandl.get('LBMA/GOLD', column_index='1', returns='numpy'))

# Standardising the dataset
metal_date, gold_usd_am = shuffle(date, gold_usd_am)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_gold_usd_am = scaler.fit_transform(gold_usd_am.reshape(-1,1))

# Creating the neural network
model = Sequential([
  Dense(units = 16, input_shape=(1,), activation='relu'),
  Dense(units = 32, activation = 'relu'),
  Dense(units = 1 , activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=metal_date, y=scaled_gold_usd_am, batch_size=10, epochs=30, shuffle=True, verbose=2)

#model.summary()

