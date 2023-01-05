#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/1 1:31
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : gd.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

f = lambda x : (x-3.5)**2 -4.5*x +10
def g(x):
    return 2*(x-3.5) -4.5
eta = 0.8

#最优解 初始化随机给个值
theta = 15
last_theta = theta + 0.1
thetas = []
'''
for i in range(50):
    theta = theta -eta * g(theta)
    print(theta)
'''
while True:
    if np.abs(last_theta - theta) < 0.001:
        break
    last_theta = theta
    theta = theta -eta *g(theta)
    thetas.append(theta)

x = np.linspace(-6,16,100)
y = f(x)
plt.figure(figsize=(9,6))
plt.plot(x,y)
thetas = np.array(thetas)
plt.scatter(thetas,f(thetas),color='red')
plt.show()
