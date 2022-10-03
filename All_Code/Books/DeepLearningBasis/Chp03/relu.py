#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/10/3 11:03
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : relu.py
# @Software: PyCharm

import numpy as np

import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0,x)

if __name__ == "__main__":
    X = np.arange(-5.0,5.0,0.1)
    y = relu(X)
    plt.plot(X,y)
    plt.ylim(-1.0,5.5)
    plt.show()