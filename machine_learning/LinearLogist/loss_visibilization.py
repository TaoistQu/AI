#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/20 17:35
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : loss_visibilization.py
# @Software: PyCharm

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

data = datasets.load_breast_cancer()
X, y = scale(data['data'][:, :2]), data['target']

lr = LogisticRegression()
lr.fit(X, y)

w1 = lr.coef_[0, 0]
w2 = lr.coef_[0, 1]

def sigmoid(X, w1, w2):
    z = w1*X[0] + w2*X[1]
    return 1/(1+np.exp(-z))

def loss_function(X, y, w1, w2):
    loss = 0
    for x_i,y_i in zip(X, y):
        p = sigmoid(x_i, w1, w2)
        loss += -1*y_i*np.log(p) - (1-y_i)*np.log(1-p)
    return loss

w1_space = np.linspace(w1-2, w1+2, 100)
w2_space = np.linspace(w2-2, w2+2, 100)

loss1_ = np.array([loss_function(X, y, i, w2) for i in w1_space])
loss2_ = np.array([loss_function(X, y, w1, i) for i in w2_space])

fig1 = plt.figure(figsize=(12, 9))
plt.subplot(2, 2, 1)
plt.plot(w1_space, loss1_)
plt.xlabel('w1 VS loss')

plt.subplot(2, 2, 2)
plt.plot(w2_space, loss2_)
plt.xlabel('w2 VS loss')

plt.subplot(2, 2, 3)
w1_grid,w2_gird = np.meshgrid(w1_space,w2_space)
loss_grid = loss_function(X,y,w1_grid,w2_gird)
plt.contour(w1_grid, w2_gird, loss_grid, 20)

plt.subplot(2, 2, 4)
plt.contourf(w1_grid, w2_gird, loss_grid, 20)
plt.savefig('./4_损失函数可视化.png', dpi=200)

fig2 = plt.figure(figsize=(12, 6))
ax = Axes3D(fig2)
ax.plot_surface(w1_grid, w2_gird, loss_grid, cmap='viridis')
plt.xlabel('w1', fontsize=20)
plt.ylabel('w2', fontsize=20)
ax.view_init(30, -30)
plt.savefig('5_损失函数可视化.png', dpi=200)
plt.show()





