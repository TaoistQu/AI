#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/28 22:51
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : formula.py
# @Software: PyCharm
import numpy as np
from sklearn.linear_model import LinearRegression

'''$$\theta = (X^TX)^{-1}X^Ty$$'''
x = np.linspace(0,12,20)+np.random.randn(20)
y = np.linspace(2,12,20)+np.random.randn(20)
ones = np.ones(shape=(20,1))
X = np.hstack((ones,x.reshape(-1,1)))

selt = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
print(selt)

linear = LinearRegression()
linear.fit(X,y)
print(linear.coef_[1],linear.intercept_)



