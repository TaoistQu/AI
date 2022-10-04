#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/4 16:21
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : gradient_1d.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np


def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)

def function1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f,x):
    d = numerical_diff(f,x)
    print(d)
    y = f(x) -d*x
    return lambda t: d*t+y


x = np.arange(0.0,20.0,0.1)

y = function1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function1,10)

y2 = tf(x)

plt.plot(x,y)
plt.plot(x,y2)
plt.show()