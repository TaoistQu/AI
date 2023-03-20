# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :padding.py
# @Time      :2023/3/20 下午9:58
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:研究CNN中的padding对卷积的影响



import keras
from keras import Sequential
from keras.layers import Input, Conv2D, MaxPool2D


#1、padding = valid  N = (W -F + 1) / S
# W 为图片长宽， F 为卷积核的尺寸 ， S为步长
model = Sequential([
    Input(shape=(8, 8, 3), dtype='float32'),
    #MaxPool2D(pool_size=3, strides=2, padding='valid'),
    keras.layers.Conv2D(64, 3, 2, padding='valid')
])

model.summary()

#1、padding = same N =  W / S
# W 为图片长宽， S为步长, 向上取整

model_smae = Sequential([
    Input(shape=(9, 11, 3), dtype='float32'),
    #MaxPool2D(pool_size=3, strides=2, padding='valid'),
    keras.layers.Conv2D(64, 3, 2, padding='same')
])

model_smae.summary()








