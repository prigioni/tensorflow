#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019-9-22 14:27 
# @Author : lauqasim
# @File : stock_predict_tf_lstm.py 
# @Software: PyCharm

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Loading dataset
tesla_stocks = pd.read_csv('tesla_stocks.csv')
data_to_use = tesla_stocks['Close'].values
print('Total number of days in the dataset: {}'.format(len(data_to_use)))

# Data preprocessing
# Scaling data
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

# timestep the dataset
X, y = create_data(scaled_dataset, 7)

# Creating Training and Testing sets
X_train = np.array(X[:700])
y_train = np.array(y[:700])
X_test = np.array(X[700:])
y_test = np.array(y[700:])
print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))


# def rnn
def LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout=True, dropout_rate=0.8):
    layer = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)

    if dropout:
        layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=dropout_rate)

    cell = tf.contrib.rnn.MultiRNNCell([layer] * number_of_layers)

    init_state = cell.zero_state(batch_size, tf.float32)

    return cell, init_state


def output_layer(lstm_output, in_size, out_size):
    x = lstm_output[:, -1, :]  # output bases on the laste output step
    weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.05), name='output_layer_weights')
    bias = tf.Variable(tf.zeros([out_size]), name='output_layer_bias')
    output = tf.matmul(x, weights) + bias
    return output


# compute loss
def opt_loss(logits, targets, learning_rate, batch_size):
    losses = []
    for i in range(targets.get_shape()[0]):
        losses.append([(tf.pow(logits[i] - targets[i], 2))])

    loss = tf.reduce_sum(losses)/(2*batch_size)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return loss, train_op


# ——————————————————定义神经网络变量——————————————————
epochs = 300
batch_size = 7
learning_rate = 0.001
hidden_layer_size = 512
number_of_layers = 1
dropout = True
dropout_rate = 0.8
number_of_classes = 1
gradient_clip_margin = 4
time_step = 7
inputs = tf.placeholder(tf.float32, [batch_size, time_step, 1], name='inputs')
targets = tf.placeholder(tf.float32, [batch_size, 1], name='targets')

def train_lstm():
    cell, init_state = LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout, dropout_rate)
    outputs, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
    logits = output_layer(outputs, hidden_layer_size, number_of_classes)
    logits = tf.identity(logits, name='logits')
    loss, opt = opt_loss(logits, targets, learning_rate, batch_size)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(epochs):
            ii = 0
            epoch_loss = []
            while (ii + batch_size) <= len(X_train):
                X_batch = X_train[ii:ii + batch_size]
                y_batch = y_train[ii:ii + batch_size]

                o, c, _ = session.run([logits, loss, opt],
                                      feed_dict={inputs: X_batch, targets: y_batch})
                epoch_loss.append(c)
                ii += batch_size
            if (i % 30) == 0:
                print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))
                # 每10步保存一次参数
                print("保存模型：", saver.save(session, 'tf.stock.model'))

train_lstm()