# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :AlexNet.py
# @Time      :2023/3/12 下午8:43
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:实现AlexNet


import tensorflow as tf
import keras
from keras.layers import Input
import pandas as pd
from keras import preprocessing

import this
def AlexNet(im_height=224, im_width=224, num_class=1000):
    input_image = Input(shape=(im_height, im_width, 3), dtype=tf.float32)
    x = keras.layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)
    x = keras.layers.Conv2D(48, kernel_size=11, strides=4, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = keras.layers.Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    #x = keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    #x = keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    #x = keras.layers.Dropout(0.2)(x)
    #x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(num_class)(x)
    predict = keras.layers.Softmax()(x)

    model = keras.models.Model(inputs=input_image, outputs=predict)
    return model


train_dir  = 'data/training/training'
valid_dir = 'data/validation/validation'
label_file ='data/monkey_labels.txt'


labels = pd.read_csv(label_file, header=0)

#生成数据
train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)


#从目录中读取图片
height = 224
width = 224
channels = 3
batch_size = 128
num_class = 10

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(height, width),
                                  batch_size=batch_size,
                                  shuffle=True,
 #                                 seed=7,
                                  class_mode='categorical')


valid_datagen = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(height, width),
                                  batch_size=batch_size,
                                  shuffle=True,
 #                                 seed=7,
                                  class_mode='categorical')

model = AlexNet(num_class=10)
summary = model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
train_num = train_generator.samples
valid_num = valid_generator.samples
histroy = model.fit(
    train_generator,
    steps_per_epoch= train_num // batch_size,
    epochs=300,
    validation_data= valid_generator,
    validation_steps= valid_num // batch_size
)

