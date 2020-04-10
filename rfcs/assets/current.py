from keras.models import Sequential
from keras.layers import TimeDistributed, SimpleRNN, Dense
from keras.utils import Sequence

import numpy as np
import math

# Assume we have recorded a large measurement with 1000 timestamps
# from a system with two inputs and one output
input_dim = 2
output_dim = 1
num_timestamps = 1000

X = np.random.randn(num_timestamps, input_dim)
Y = np.random.randn(num_timestamps, output_dim)

# prepare data for training on sequences consisting of
# 100 timestamps
T = 100
n_samples = num_timestamps//T

assert n_samples == 10

# batch consists of 3 sequences of length T
batch_size = 3
n_batches = math.ceil(n_samples / batch_size)

assert n_batches == 4

# preallocate training matrices
X_train = np.zeros((n_batches, batch_size, T, input_dim))
Y_train = np.zeros((n_batches, batch_size, T, output_dim))

# fill data into training matrices
for b in range(n_batches):
    batch_start_idx = b*batch_size*T
    batch_end_idx = min((b+1)*batch_size*T, num_timestamps)
    num_samples = (batch_end_idx-batch_start_idx)//T
    X_train[b] = X[batch_start_idx:batch_end_idx, :].reshape((num_samples, T, input_dim))
    Y_train[b] = Y[batch_start_idx:batch_end_idx, :].reshape((num_samples, T, output_dim))

# custom sequence class
class CustomSequence(Sequence):

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

cs = CustomSequence(X_train, Y_train)

# define many-to-many model
model = Sequential()
model.add(SimpleRNN(5, return_sequences=True,
                    input_shape=(T, input_dim)))
model.add(TimeDistributed(Dense(output_dim, activation="linear")))

model.compile(optimizer="sgd", loss="mse")
model.fit(cs, epochs=2)
