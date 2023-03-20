# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :GoogleNet.py
# @Time      :2023/3/13 下午9:06
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:

import tensorflow as tf
import keras
from keras.layers import Layer, Conv2D, MaxPool2D, AvgPool2D, Dense, Softmax, Flatten, Dropout, Input
from keras import Sequential
from keras.models import Model
import numpy as np

import matplotlib.pyplot as plt

class Inception(Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super().__init__(**kwargs)
        self.branch1 = Conv2D(ch1x1, kernel_size=1, activation='relu')
        self.branch2 = Sequential([
            Conv2D(ch3x3red, kernel_size=1, activation='relu'),
            Conv2D(ch3x3, kernel_size=3, padding='SAME', activation='relu')
        ])
        self.branch3 = Sequential([
            Conv2D(ch5x5red, kernel_size=1, activation='relu'),
            Conv2D(ch5x5, kernel_size=5, padding='SAME', activation='relu')
        ])

        self.branch4 = Sequential([
            MaxPool2D(pool_size=3, strides=1, padding='SAME'),
            Conv2D(pool_proj, kernel_size=1, activation='relu'),
        ])

    def call(self, inputs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)

        outputs = keras.layers.concatenate([branch1, branch2, branch3, branch4])

        return outputs



#辅助输出结构
class InceptionAux(Layer):
    def __init__(self, num_classes, **kwargs):
        self.average_pool = AvgPool2D(pool_size=5, strides=3)
        self.conv = Conv2D(128, kernel_size=1, activation='relu')
        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(num_classes)
        self.softmax = Softmax()

    def call(self, inputs):
        x = self.average_pool(inputs)
        x = self.conv(x)
        x = Flatten()(x)
        x = Dropout(rate=0.5)(x)
        x = self.fc1(x)
        x = Dropout(rate=0.5)(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

def GooLeNet(im_height=224, im_width=224, num_classes=1000, aux_logist=False):
    input_image = Input(shape=(im_height, im_width, 3), dtype='float32')
    x = Conv2D(64, kernel_size=7, activation='relu', padding='SAME', strides=2)(input_image)
    x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)
    x = Conv2D(64, kernel_size=1, strides=1, activation='relu', padding='SAME')(x)
    x = Conv2D(192, kernel_size=3, strides=1, activation='relu', padding='SAME')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    x = Inception(64, 96, 128, 16, 32, 32, name='inception_3a')(x)
    x = Inception(128, 128, 192, 32, 96, 64, name='inception_3b')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    x = Inception(192, 96, 208, 16, 48, 64, name='inception_4a')(x)
    if aux_logist:
        aux1 = InceptionAux(num_classes, name='aux_1')(x)
    x = Inception(160, 112, 224, 24, 64, 48, 64, name='inception_4b')(x)
    x = Inception(128, 128, 256, 24, 64, 64, name='inception_4c')(x)
    x = Inception(112, 144, 288, 32, 64, 64, name='inception_4d')(x)
    if aux_logist:
        aux2 = InceptionAux(num_classes, name='aux_2')(x)

    x = Inception(256, 160, 320, 32, 128, 128, name='inception_4e')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    x = Inception(256, 160, 320, 32, 128, 128, name='inception_5a')(x)
    x = Inception(384, 192, 384, 48, 128, 128, name='inception_5b')(x)

    x = AvgPool2D(pool_size=7, strides=1)(x)
    x = Flatten()(x)
    x = Dropout(rate=0.4)

    x = Dense(num_classes)(x)
    aux3 = keras.layers.Softmax()(x)

    if aux_logist:
        model = Model(inputs=input_image, outputs=[aux1, aux2, aux3])
    else:
        model = Model(inputs = input_image, outputs=aux3)

    return model





