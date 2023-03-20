# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :Vgg.py
# @Time      :2023/3/13 下午8:05
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:

import tensorflow as tf

import keras
from keras.layers import MaxPool2D,Conv2D,Input,Flatten,Dense,Dropout,Softmax
from keras import Sequential
import numpy as np
import pandas as pd
from keras import preprocessing

import matplotlib.pyplot as plt

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_feature(cfg):
    feature_layers = []
    for v in cfg:
        if v == 'M':
            feature_layers.append(MaxPool2D())
        else:
            feature_layers.append(Conv2D(v, kernel_size=3, padding='SAME', activation='relu'))
    return Sequential(feature_layers, name='feature')

def VGG(feature, im_height=244, im_width=244, num_classes =1000):
    input_image = Input(shape=(im_height, im_width, 3), dtype='float32')
    x = feature(input_image)
    x = Flatten()(x)

    x = Dense(2048, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(num_classes)(x)
    output = Softmax()(x)

    model = keras.models.Model(inputs=input_image,outputs=output)
    return model

def vgg(model_name = 'vgg11' ,im_height=224, im_width=224, num_classes=1000):
    cfg = cfgs[model_name]
    model = VGG(make_feature(cfg), im_height,im_width,num_classes=num_classes)

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
batch_size = 64
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

vgg_16 = vgg(num_classes=10)
summary = vgg_16.summary()
vgg_16.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
train_num = train_generator.samples
valid_num = valid_generator.samples
histroy = vgg_16.fit(
    train_generator,
    steps_per_epoch= train_num // batch_size,
    epochs=30,
    validation_data= valid_generator,
    validation_steps= valid_num // batch_size
)



