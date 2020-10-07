import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

from random import randint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.python.eager.context import PhysicalDevice
from tensorflow.python.framework.config import list_physical_devices

train_labels = []
train_samples = []

for i in range(50):

  random_younger = randint(13,64)
  train_samples.append(random_younger)
  train_labels.append(1)

  random_older = randint(65,100)
  train_samples.append(random_older)
  train_labels.append(0)

for i in range(1000):

  random_younger = randint(13,64)
  train_samples.append(random_younger)
  train_labels.append(0)

  random_older = randint(65,100)
  train_samples.append(random_older)
  train_labels.append(1)

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print(len(physical_devices))

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))


# this is the neural network. units = nodes, input shape needs to be initalized before the this class is defined, softmax gives probability of the outputs
model = Sequential([
  Dense(units = 16, input_shape=(1,), activation='relu'),
  Dense(units = 32, activation = 'relu'),
  Dense(units = 2 , activation='softmax')
])

# shows model summary
#model.summary()

# epochs means how many times the network will run over the data
# data should be shuffled to removed the order
#verbose will show an output

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=True, verbose=2)
