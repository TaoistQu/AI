#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/2 1:03
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : bgd.py
# @Software: PyCharm

import numpy as np
X = np.random.rand(100,3)
#print(X)
#生成一个斜率和截距
W = np.random.randint(1,10,size=(3,1))
b = np.random.randint(1,10,size=1)
y = X.dot(W) + b + np.random.randn(100,1)
#c_将两个数据级联,默认列合并
X = np.c_[X,np.ones(shape=(100,1))]
#等价于
#X = np.concatenate([X,np.ones(shape=(100,1))],axis=1)

#训练次数
epoches = 20000

#变学习率
t0,t1 = 5,1000
def learning_rate_schedule(t):
    return t0/(t+t1)

#初始化w0,w1....wn
theta = np.random.randn(4,1)

#for 循环操作
#加条件判断是否收敛
loss = 0 #损失函数的值
loss_last = loss +1

for i in range(epoches):
    if np.abs((loss_last - loss)) < 0.0001:
        print('执行多少次%d退出！'%(i))
        break
    g = X.T.dot(X.dot(theta) - y)
    learning_rate = learning_rate_schedule(i)

    theta = theta - learning_rate *g
    loss_last = loss
    loss = ((X.dot(theta) - y)**2).sum()

print('真是的斜率和截距是：\n',W,b)
print('BGD梯度下降结果是：\n',theta)
'''
plt.scatter(X[:,0],y)
x = np.linspace(0,1,100)
y = x * theta[0,0] + theta[1,0]
plt.plot(x,y,color = 'green')
plt.show()
'''