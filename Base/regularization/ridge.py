#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/11 13:09
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : ridge.py
# @Software: PyCharm
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

X = 2*np.random.rand(100,5)
w = np.random.randint(1,10,size=(5,1))
b = np.random.randint(1,10,size=1)
y = X.dot(w) + b + np.random.randn(100,1)

print('原始方程的斜率：',w.ravel())
print('原始方程的截距：',b)

ridge = Ridge(alpha=1,solver='sag')
ridge.fit(X,y)
print('岭回归求解的斜率：',ridge.coef_)
print('岭回归求解的截距：',ridge.intercept_)

#线性回归梯度下降法
sgd = SGDRegressor(penalty='l2',alpha=0,l1_ratio=0)
sgd.fit(X,y.reshape(-1,))
print('随机梯度下降求解的斜率是：',sgd.coef_)
print('随机梯度下降求解的截距是：',sgd.intercept_)


