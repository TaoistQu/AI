#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/6 16:33
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : layers.py
# @Software: PyCharm
import numpy as np
from Books.DeepLearningBasis.common.functions import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self,dout):
        dx = dout*(1.0 - self.out)*self.out   # dx = dout * y * (1-y)

        return dx


