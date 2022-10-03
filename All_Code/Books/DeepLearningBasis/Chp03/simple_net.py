#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/3 13:54
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : simple_net.py
# @Software: PyCharm

import numpy as np
from sigmoid import sigmoid

def identity_function(x):
    return x

def init_network():
    network = {};
    network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['w2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['w3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network,x):
    W1,W2,W3 = network['w1'],network['w2'],network['w3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = identity_function(a3)

    return y

if __name__ == "__main__":
    network = init_network()
    X = np.array([1.0,0.5])
    y = forward(network,X)

    print(y)

