#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/3 23:37
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : neuralnet_mnist.py
# @Software: PyCharm
import os
import sys
from dataset import mnist
from common import functions
import numpy as np
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f :
        network = pickle.load(f)
    return network

def predict(network,x):
    W1 , W2, W3 =  network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = functions.sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = functions.sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = functions.softmax(a3)

    return y

x , t = get_data()
network = init_network()
accuracy_cnt = 0

for i in range(len(x) ):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))



