# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :data_sets.py
# @Time      :2023/3/5 下午9:19
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description: 包装自己的数据为tensorflow的数据集

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
dataset = dataset.repeat(3).batch(7)
for batch in dataset:
    print(batch)


#从元组创建dataset(x, y)
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'frog'])

dataset = tf.data.Dataset.from_tensor_slices((x, y))

for item_x, item_y in dataset:
    print(item_x.numpy(), item_y.numpy().decode())

#从字典创建
dataset = tf.data.Dataset.from_tensor_slices(
    {'feature': x,
     'label': y}
)

for item in dataset:
    print(item['feature'].numpy(), item['label'].numpy().decode())


#tensorflow数据集最常用的一种用法:把文件名定义为dataset --> 读文件 --> 变成具体的数据集

dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
dataset = dataset.repeat(3).batch(7)
dataset = dataset.interleave(lambda v : tf.data.Dataset.from_tensor_slices(v),
                   cycle_length=4,
                   block_length=5)
for item in dataset:
    print(item)