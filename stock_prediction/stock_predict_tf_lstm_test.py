#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019-9-22 15:08 
# @Author : lauqasim
# @File : stock_predict_tf_lstm_test.py 
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

tesla_stocks = pd.read_csv('tesla_stocks.csv')
data_to_use = tesla_stocks['Close'].values
print('Total number of days in the dataset: {}'.format(len(data_to_use)))

scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(data_to_use.reshape(-1, 1))


def create_data(data, time_step):
    X = []
    y = []
    for i in range(len(data) - time_step):
        a = data[i:i+time_step]
        X.append(a)
        y.append(data[i+time_step])
    assert len(X) == len(y)
    return X, y

X, y = create_data(scaled_dataset, 7)
X_train = np.array(X[:700])
y_train = np.array(y[:700])
X_test = np.array(X[700:])
y_test = np.array(y[700:])

load_path = 'tf.stock.model'
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as session:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(session, load_path)
    inputs = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    tests = []
    i = 0
    batch_size = 7
    while i + batch_size <= len(X_test):
        o = session.run([logits], feed_dict={inputs: X_test[i:i + batch_size]})
        i += batch_size
        tests.append(o)
    tests_new = []
    for i in range(len(tests)):
        for j in range(len(tests[i][0])):
            tests_new.append(tests[i][0][j])
    plt.figure(figsize=(16, 7))
    plt.plot(y_test[:-1], label='y_test', color='r')
    plt.plot(tests_new[1:], label='y_predict', color='b')
    plt.legend()
    plt.show()