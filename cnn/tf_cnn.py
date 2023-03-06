# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :tf_cnn.py
# @Time      :2023/3/6 下午8:57
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

moon = plt.imread('./images/moonlanding.png')
cat = plt.imread('images/cat.jpg')
cat = cat.mean(axis=2)
plt.figure(figsize=(10, 8))
plt.imshow(cat, cmap='gray')
plt.show()
print(cat.shape)
#均值滤波
input_img = tf.constant(cat.reshape(1, 456, 730, 1 ),dtype=tf.float32)
#均值核滤波
#filters = tf.constant(np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).reshape(3, 3, 1, 1), dtype=tf.float32)
#高斯滤波
filters = tf.constant(np.array([[0, 1, 0], [0, -4, 1], [0, 1, 0]]).reshape(3, 3, 1, 1), dtype=tf.float32)
strides = [1, 1, 1, 1]
conv2d = tf.nn.conv2d(input_img,filters=filters, strides=strides, padding='SAME')
plt.figure(figsize=(10, 8))
plt.imshow(conv2d.numpy().reshape(456, 730), cmap='gray')
plt.show()

#彩色图片的卷积,把彩色图片当成多个单通道的图片处理

euro = plt.imread('images/欧式.jpg')
plt.imshow(euro)
plt.show()
input_euro = tf.constant(euro.reshape(1, 582, 1024, 3).transpose(3, 1, 2, 0), dtype=tf.float32)
#高斯滤波
filters = filters = tf.constant(np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).reshape(3, 3, 1, 1), dtype=tf.float32)
strides = [1, 1, 1, 1]
conv2d = tf.nn.conv2d(input_euro, filters=filters, strides=strides, padding='SAME')
plt.figure(figsize=(10, 8))
plt.imshow(conv2d.numpy().reshape(3, 582, 1024).transpose(1, 2, 0)/ 255.0)
plt.show()


