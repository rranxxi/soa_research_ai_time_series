import time
import math

import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from vis import *


def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape
    
    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))
    
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))
    
    return X_train, X_test

def preprocess_data(raw_data, col_names):
    scale = StandardScaler().fit(raw_data)
    proc_dat = scale.transform(raw_data)
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(raw_data.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False
    X = proc_dat[:, mask]
    y = proc_dat[:, ~mask]
    return X, y, scale


def visualize_index(raw_data, scaled_data):
    plt.plot(raw_data['NDX'])
    plt.title('Nasdaq Index Jul 26 to Dec 22 2016')
    plt.subplot(1,2,1)
    plt.plot(raw_data['NDX'], '-', color='mediumvioletred', label='Raw')
    plt.legend(loc='upper left')
    plt.subplot(1,2,2)
    plt.plot(scaled_data, color='royalblue', label='Normalized')
    plt.legend(loc='upper left')
    plt.show()



def build_model(window, input_size, output_size = 1):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        units = 64, 
        input_shape = (window, input_size),
        return_sequences=True))
    
    model.add(Dropout(0.4))

    model.add(LSTM(
        64,
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=output_size))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


NASDAQ = '~/LSTM-Stock-Pred/nasdaq/data/nasdaq/small/nasdaq100_padding.csv'

batch_size = 128
time_window = 20
n_epochs = 20
train_test_split = 0.7

raw_data = pd.read_csv(NASDAQ)
X, y, scale = preprocess_data(raw_data, ['NDX'])
train_size = int(X.shape[0] * train_test_split)


# Build a model
model = build_model(window = time_window - 1, input_size = X.shape[1], output_size = 1)
model.summary()

# Train the model
loss_epoches = []
for i in tqdm(range(n_epochs)):
    perm_idx = np.random.permutation(train_size - time_window)
    
    sum_loss = 0
    for j in range(0, train_size, batch_size):    
        batch_idx = perm_idx[j:(j + batch_size)]

        X_batch = np.zeros((len(batch_idx), time_window - 1, X.shape[1]))
        y_history = np.zeros((len(batch_idx), time_window - 1, 1))
        y_target = y[batch_idx + time_window]

        for k in range(len(batch_idx)):
            X_batch[k, :, :] = X[batch_idx[k] : (batch_idx[k] + time_window - 1), :]
            y_history[k, :] = y[batch_idx[k] : (batch_idx[k] + time_window - 1)]
        loss =  model.train_on_batch(X_batch, y_target)[0]
    loss_epoches.append(loss)
    

total_size = X.shape[0]
batch_index = range(0, total_size - time_window)

y_pred = []
# Predict 
for j in range(0, total_size, batch_size):     
    batch_idx = batch_index[j:(j + batch_size)]
    X_batch = np.zeros((len(batch_idx), time_window - 1, X.shape[1]))
    y_history = np.zeros((len(batch_idx), time_window - 1, 1))

    for k in range(len(batch_idx)):
        X_batch[k, :, :] = X[batch_idx[k] : (batch_idx[k] + time_window - 1), :]

    y_pred_batch = model.predict_on_batch(X_batch)
    for yy in y_pred_batch:
        y_pred.append(yy)

# Visualize the results
plt.plot(range(0, total_size), y, '-c', lw=3, label = 'index ground_truth')
plt.plot(range(0, train_size), y_pred[0:train_size], '-b', label = 'index train')
plt.plot(range(train_size, len(y_pred)), y_pred[train_size:], '-r', label = 'index test')
plt.legend(loc='best')

plt.title('Model Nasdaq Prediction Results')
plt.xlabel('time')
plt.ylabel('Normalize Nasdaq 100 Index')
plt.show()

