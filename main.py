from random import randint

import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScalar
from sklearn.utils import shuffle

train_labels = []
train_samples = []

for i in range(50):

  random_younger = randint(13,64)
  train_labels.append(1)

  random_older = randint(65,100)
  train_samples.append(random_younger)
  train_labels.append(0)

for i in range(1000):

  random_younger = randint(13,64)
  train_samples.append(random_younger)
  train_labels.append(0)

  random_older = randint(65,100)
  train_samples.append(random_younger)
  train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScalar(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

for i in scaled_train_samples:
    print(i)
