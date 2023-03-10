# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :monkey_10.py
# @Time      :2023/3/9 下午9:56
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras import preprocessing
import numpy as np
import pandas as pd

train_dir  = 'data/training/training'
valid_dir = 'data/validation/validation'
label_file ='data/monkey_labels.txt'

labels = pd.read_csv(label_file, header=0)

#生成数据
train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range= 40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

#从目录中读取图片
height = 256
width = 256
channels = 3
batch_size = 32
num_class = 10

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(height, width),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  seed=7,
                                  class_mode='categorical')


valid_datagen = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(height, width),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  seed=7,
                                  class_mode='categorical')


#创建model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, padding='same',
                 activation='relu', input_shape=(256, 256, 3)))

#池化
model.add(MaxPool2D())

model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                 activation='relu'))

model.add(MaxPool2D())

model.add(Conv2D(filters=128, kernel_size=3, padding='same',
                 activation='relu'))

model.add(MaxPool2D())

model.add(Conv2D(filters=256, kernel_size=3, padding='same',
                 activation='relu'))

model.add(MaxPool2D())


model.add(Conv2D(filters=256, kernel_size=3, padding='same',
                 activation='relu'))

model.add(MaxPool2D())
#flatten处理
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
summary = model.summary()
train_num = train_generator.samples
valid_num = valid_generator.samples
model.fit(
    train_generator,
    steps_per_epoch= train_num // batch_size,
    epochs=60,
    validation_data= valid_generator,
    validation_steps= valid_num // batch_size
)