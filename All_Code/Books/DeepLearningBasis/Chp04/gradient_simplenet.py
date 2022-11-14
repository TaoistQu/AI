#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/4 23:46
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : gradient_simplenet.py
# @Software: PyCharm
import numpy as np
import os
import sys

from common.gradient import numerical_gradient
from common.functions import softmax, cross_entropy_error


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w : net.loss(x,t)
dW = numerical_gradient(f,net.W)
print(dW)
