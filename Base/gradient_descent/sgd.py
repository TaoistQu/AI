#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/4 23:31
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : sgd.py
# @Software: PyCharm
import numpy as np
#1、创建数据集X,y
X = 2*np.random.rand(100,1)
w,b = np.random.randint(1,10,size=2)
y = w*X + b + np.random.randn(100,1)

#2、使用偏置项 x_0 = 1,更新X
X = np.c_[X,np.ones((100,1))]

#3、创建超参数轮次、样本数量
epochs = 10000

#4、定义一个学习率调整函数
t0,t1 = 5,500
def learning_rate_schedule(t):
    return t0 / (t+t1)

#5、初始化 w0....wn
theta = np.random.randn(2,1)

#6、多次for循环实现梯度下降，最终结果收敛
#条件判断，是否收敛
loss = 0
loss_last = loss +1

for epoch in range(epochs):
    if np.abs(loss - loss_last) < 0.0001:
        print('执行%d次后推出'%(epoch))
        break
    #在双层for循环中，每一轮次开始分批次迭代之前打乱的数据索引顺序
    index = np.random.randint(0,100,size=1)[0]
    X_i = X[[index]]
    y_i = y[[index]]

    g = X_i.T.dot(X_i.dot(theta)-y_i)
    learning_rate = learning_rate_schedule(epoch)
    theta = theta - g*learning_rate

    loss_last = loss
    loss = ((X.dot(theta)-y)**2).sum()

print('真是的斜率和截距是：\n',w,b)
print('BGD梯度下降结果是：\n',theta)