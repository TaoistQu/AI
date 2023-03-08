# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :baseline.py
# @Time      :2023/3/8 下午8:17
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:使用神经网络实现CNN

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train.reshape(55000, -1) / 255.0)
x_valid_scaled = scaler.transform(x_valid.reshape(5000, -1) / 255.0)
x_test_scaled = scaler.transform(x_test.reshape(10000, -1) / 255.0)

def make_dataset(data, target, epochs, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((data, target))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size).prefetch(50)

    return dataset

batch_size = 64
epochs = 30
train_dataset = make_dataset(x_train_scaled, y_train, epochs, batch_size)

#创建model
model = keras.models.Sequential()

model.add(Dense(512, activation='relu', input_shape=(784, )))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
summary = model.summary()
print(summary)

eval_dataset = make_dataset(x_valid_scaled, y_valid, epochs=1, batch_size=32, shuffle=False)

history = model.fit(train_dataset,
                    steps_per_epoch=x_train_scaled.shape[0] // batch_size,
                    epochs=30,
                    validation_data=eval_dataset)
test_dataset = make_dataset(x_test_scaled, y_test, 1, 32)
model.evaluate(test_dataset)