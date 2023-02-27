# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :dropout.py
# @Time      :2023/2/22 下午11:40
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description: wide_deep

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
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(32, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(32, activation='relu')(hidden1)

concat = keras.layers.concatenate([input_wide, hidden2])
#这个输出为wide——deep的输出
output = keras.layers.Dense(1)(concat)

#单独把deep的输出拿出来
output2 = keras.layers.Dense(1)(hidden2)

##包装成一个model
model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output, output2])

summary = model.summary()
#配置训练
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])
#存放tensorboard的日志路径
log_dir = r'/media/ql/hdd/code/AI/DP/callbacks'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
output_model_file = os.path.join(log_dir, 'model.h5')
#常用的回调，有三种
# 1、tensorboard可视化工具
#2、modelcheckpoint
#3、earlyStopping 早停止法
x_train_scaled_wide = x_train_scaled[:, :5]
x_train_scaled_deep = x_train_scaled[:, 2:]
x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]
x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]

history = model.fit([x_train_scaled_wide, x_train_scaled_deep], y_train, batch_size=64, epochs=100, validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], y_valid))
#model.evaluate(x_valid_scaled, y_valid)
pd.DataFrame(history.history).plot()
plt.show()


