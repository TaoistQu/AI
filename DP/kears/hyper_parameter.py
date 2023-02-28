# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :call_back.py
# @Time      :2023/2/26 上午11:26
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description: 超参数搜素
# 一般的超参数 有学习率
import os
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


housing = fetch_california_housing()
data = housing.data
target = housing.target

x_train_all, x_test, y_train_all, y_test = train_test_split(data, target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all)


#df = pd.DataFrame(x_train, columns=housing.feature_names)

#正态标准化
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#定义网络
'''
model = Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)
])

summary = model.summary()
'''
#配置训练

#存放tensorboard的日志路径
'''
log_dir = r'/media/ql/hdd/code/AI/DP/callbacks'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
output_model_file = os.path.join(log_dir, 'model.h5')
'''
#常用的回调，有三种
# 1、tensorboard可视化工具
#2、modelcheckpoint
#3、earlyStopping 早停止法
'''
callbacks = [
    keras.callbacks.TensorBoard(log_dir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3) #5次保持不变
]
'''
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

historys = []
for lr in learning_rates:
    model = Sequential(
        [Dense(32, activation='relu', input_shape=x_train.shape[1:]),
         Dense(1)]
    )
    optimizer = SGD(lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)  # 5次保持不变
    ]
    history = model.fit(x_train_scaled, y_train, batch_size=64, epochs=100, validation_data=(x_valid_scaled, y_valid), callbacks=callbacks)
    #model.evaluate(x_valid_scaled, y_valid)
    historys.append(history)

def plot_learning_curve(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid()
    plt.gca().set_ylim(0, 1)
    plt.show()

for lr, history in zip(learning_rates, historys):
    print(lr)
    plot_learning_curve(history)


