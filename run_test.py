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


gold = quandl.get("LBMA/GOLD")
np.array(gold)
print(gold[:,0])
