#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/4 23:51
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : ngdb.py
# @Software: PyCharm

import numpy as np
X = 2*np.random.rand(100,5)
w = np.random.randint(1,10,(5,1))
b = np.random.randint(1,10,1)
y = X.dot(w) + b + np.random.randn(100,1)

X = np.c_[X,np.ones((100,1))]

epochs = 10000

t0,t1 = 5,5000
def learning_rate_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(6,1)

loss = 0
loss_last = loss+1

for epoch in range(epochs):
    if np.abs(loss - loss_last) < 0.000001:
        print('执行%d次退出!'%(epoch))
        break
    index = np.random.randint(0,100,size=1)[0]
    X_i = X[[index]]
    y_i = y[[index]]

    g = X_i.T.dot(X_i.dot(theta) - y_i)

    learning_rate = learning_rate_schedule(epoch)
    theta = theta - g*learning_rate

    loss_last = loss
    loss = ((X.dot(theta)-y)**2).sum()

print('真是的斜率和截距是：\n',w,b)
print('BGD梯度下降结果是：\n',theta)