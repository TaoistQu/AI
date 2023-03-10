# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :baseline.py
# @Time      :2023/3/8 下午8:17
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:使用神经网络实现CNN


import tensorflow as tf
import keras
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler


(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train.reshape(45000, -1) / 255.0).reshape(-1, 32, 32, 3)
x_valid_scaled = scaler.transform(x_valid.reshape(5000, -1) / 255.0).reshape(-1, 32, 32, 3)
x_test_scaled = scaler.transform(x_test.reshape(10000, -1) / 255.0).reshape(-1, 32, 32, 3)

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
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, padding='same',
                 activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=32, kernel_size=3, padding='same',
                 activation='relu'))

#池化
model.add(MaxPool2D())

model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                 activation='relu'))

model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                 activation='relu'))

model.add(MaxPool2D())

model.add(Conv2D(filters=128, kernel_size=3, padding='same',
                 activation='relu'))

model.add(Conv2D(filters=128, kernel_size=3, padding='same',
                 activation='relu'))

model.add(MaxPool2D())
#flatten处理
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
summary = model.summary()

eval_dataset = make_dataset(x_valid_scaled, y_valid, epochs=1, batch_size=32, shuffle=False)

history = model.fit(train_dataset,
                    steps_per_epoch=x_train_scaled.shape[0] // batch_size,
                    epochs=30,
                    validation_data=eval_dataset)

test_dataset = make_dataset(x_test_scaled, y_test, 1, 32)
model.evaluate(test_dataset)