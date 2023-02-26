# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :dropout.py
# @Time      :2023/2/22 下午11:40
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description: wide_deep使用子类来实现

import os
import keras

import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

print(datasets.get_data_home())

housing = fetch_california_housing()
data = housing.data
target = housing.target

x_train_all, x_test, y_train_all, y_test = train_test_split(data, target, random_state=7)
x_train, x_valid, y_train,y_valid = train_test_split(x_train_all, y_train_all)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

df = pd.DataFrame(x_train, columns=housing.feature_names)
print(df.info)
print(df.describe())
#正态标准化
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#定义网络
#wide&deep模型就不能用Sequential来写
#函数式API写法：即每一层都成一个函数来用

class WideDeepModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = keras.layers.Dense(32, activation='relu')
        self.hidden2 = keras.layers.Dense(32, activation='relu')
        self.output_layer = keras.layers.Dense(1)

    def call(self, input):
        #正常传播会调用call方法
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_layer(concat)
        return output



##包装成一个model
model = WideDeepModel()
model.build(input_shape=[None, 8])

summary = model.summary()
#配置训练
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])

history = model.fit(x_train_scaled, y_train, batch_size=64, epochs=100, validation_data=(x_valid_scaled, y_valid))
model.evaluate(x_valid_scaled, y_valid)
pd.DataFrame(history.history).plot()
plt.show()


