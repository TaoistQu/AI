#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/3 23:50
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : functions.py
# @Software: PyCharm

import numpy as np

def identity_function(x):
    return x

def step_function(x):
    return np.array(x>0,dtype=np.int32)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x,axis=0)
        y = np.exp(x) / np.sum(np.exp(x),axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


