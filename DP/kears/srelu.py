# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :dropout.py
# @Time      :2023/2/22 下午11:40
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description: srelu作为激活函数

import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import AlphaDropout
from sklearn.preprocessing import StandardScaler

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
#正标准化
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

#one hot

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
#统计数据的分布
#plt.hist(X_test_scaled, bins=30, range=[-2.5, 2.5])
#

#定义网络

model = keras.Sequential()
#加dropout正则

model.add(Dense(64, activation='selu', input_shape=(784,)))

#model.add(Dropout(0.2))
#AlphaDropout 1、保持数据的方差和均值不变，既保持数据分布。2、归一化性质不变
#model.add(AlphaDropout(0.2))
model.add(Dense(64, activation='selu'))

#model.add(Dropout(0.2))
#model.add(AlphaDropout(0.2))
model.add(Dense(10, activation='softmax'))

summary = model.summary()
print(summary)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train_scaled, y_train, batch_size=64, epochs=20, validation_data=(X_test_scaled, y_test))
print(history.history)

pd.DataFrame(history.history).plot()
plt.show()
