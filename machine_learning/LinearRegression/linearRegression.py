#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/28 1:35
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : linearRegression.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn.linear_model import LinearRegression

x = np.linspace(0, 10, 20) +np.random.randn(20)
y = np.linspace(0, 12, 20) + np.random.randn(20)

plt.scatter(x, y)
plt.show()

linear = LinearRegression()
linear.fit(x.reshape(-1, 1), y)
w = linear.coef_
display(w)
b = linear.intercept_
display(b)

plt.scatter(x,y)
plt.plot(x, w*x+b, c='r')
plt.show()

x_test = np.linspace(2,8,50)
y_ = linear.predict(x_test.reshape(-1,1))
plt.scatter(x,y)
plt.plot(x_test, y_ ,c='g')
plt.show()




