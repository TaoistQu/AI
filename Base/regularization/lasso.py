#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/10 22:55
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : lasso.py
# @Software: PyCharm
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor


X = 2*np.random.rand(100,20)
w = np.random.randn(20,1)
b = np.random.randint(1,10,size=1)
y = X.dot(w) + b + np.random.randn(100,1)

print('原始方程的斜率：',w.ravel())
print('原始方程的截距：',b)

lasso = Lasso(alpha=0.1)
lasso.fit(X,y)
print('lasso回归求解的斜率：',lasso.coef_)
print('lasso回归求解的截距：',lasso.intercept_)

#线性回归梯度下降求解的结果
sgd = SGDRegressor(penalty='l1',alpha=0,eta0=0.05,max_iter=10000,learning_rate='constant')
sgd.fit(X,y.reshape(-1,))
print('随机梯度下降求解的斜率是：',sgd.coef_)
print('随机梯度下降求解的截距是：',sgd.intercept_)
