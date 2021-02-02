# Packages
import quandl
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from numpy import *
from matplotlib.pyplot import *

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

# Getting mid values the open and close data
while (count < len(gold_usd_open)):
    temp = (gold_usd_open[count] + gold_usd_close[count]) / 2.0 # Typical mid poin between two data entries

    if (gold_usd_open[count] == 0 or gold_usd_close[count] == 0):
        mid_prices.append(temp * 2.0) # if one data entries for either morn or even is not entered the data entry that was entered will be taken

    else:
        mid_prices.append(temp)

    count = count + 1

count = 0

# rare case if both data entries are 0 then the previous and next data entry will act as the data points
while(count < len(mid_prices)-1):
    if (mid_prices[count] == 0):
        temp = (mid_prices[count-1] + mid_prices[count+1]) # Get mid point between the two data entries
        if(temp == mid_prices[count-1] or temp == mid_prices[count+1]): mid_prices[count] = temp # Checks if one the data entries is 0
        else: mid_prices[count] = temp/2.0

    count = count + 1

# Problem with this is that if we have data entries [x, y , z]
# and the y is 0 then we would have to that

''' Preparing Data'''
# Splitting training and testing data set
unscaled_train_data = mid_prices[:13000]
unscaled_test_data = mid_prices[13000:]

# Normalising Data
scaler = MinMaxScaler()
train_data = np.array(unscaled_train_data).reshape(-1, 1)
test_data = np.array(unscaled_test_data).reshape(-1, 1)

# Smoothing
smoothing_window_size = 3000
for di in range(0, 12000, smoothing_window_size):
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

for ti in range(13000):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA

# Combining data back together
all_mid_data = np.concatenate([train_data, test_data], axis=0)

class DataGeneratorSeq(object):

    def __init__(self,prices,batch_size,num_unroll): # Initial variables
        self._prices = prices # Price data
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size # Size of array (How many data entries in each point)
        self._num_unroll = num_unroll
        self._segments = self._prices_length//self._batch_size # Number of batchs a data set will have
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        # Changed all values in the array to 0 (Removes all nans)
        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)

        # batches are the groups of data from the train data to be used
        # to train the network.
        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length: self._cursor[b] = np.random.randint(0,(b+1)*self._segments)

            # Stores the input data and labels which is randomized
            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length # Finds curser point for each batch of data

        return batch_data,batch_labels

    def unroll_batches(self):
        # Defines arrays
        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length-1))



dgs = DataGeneratorSeq(train_data,5,5)
ur_data, ur_labels = dgs.unroll_batches()

'''
# Enumerate keep track like a counter
# ur_data and ur_labels will be zipped together as pairs in a 2d array
# This makes it easier to have the input and output next to each other
for ur_index,(ur_data_value,ur_label_value) in enumerate(zip(ur_data,ur_labels)):
    print('\n\nUnrolled index %d'%ur_index)
    print('\tInputs: ',ur_data_value )
    print('\n\tOutput:',ur_label_value)
'''


D = 1 # Dimensionality of the data
num_unrolling = 50 # Number of time steps you look into the future.
batch_size = 500 # Number of data entries in the each batch
num_nodes = [200,200,150] # Number of hidden nodes in each layer
n_layers = len(num_nodes) # number of layers
dropout = 0.2 # dropout amount

tf.compat.v1.reset_default_graph() # Clears the default graph stack and resets the global default graph. 
# This is important in case you run this multiple times

# Input data.
train_inputs, train_outputs = [],[]
tf.compat.v1.disable_eager_execution()

# You unroll the input over time defining placeholders for each time step
for ui in range(num_unrolling):
    train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=(batch_size,D),name='train_inputs_%d'%ui))
    train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=(batch_size,1), name ='train_outputs_%d'%ui))

print("done")
# https://www.datacamp.com/community/tutorials/lstm-python-stock-market
