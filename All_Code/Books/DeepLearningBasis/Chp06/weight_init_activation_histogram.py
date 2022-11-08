#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/17 0:33
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : weight_init_activation_histogram.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def ReLu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000,100)
node_num = 100

hidden_layer_size = 5
activations = {}

x = input_data
for i in range(hidden_layer_size):
    if i!= 0:
        x = activations[i-1]
    #w = np.random.randn(node_num,node_num)*1

    #w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x,w)

    #z = sigmoid(a)
    #z = ReLu(a)
    z = tanh(a)

    activations[i] = z

for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    if i!= 0 :plt.yticks([],[])
    #plt.xlim(0.1,1)
    #plt.ylim(0,700)
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()

