# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :linear_regression.py
# @Time      :2023/3/26 下午10:13
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

income = pd.read_csv('data/Income1.csv')


#定义预测函数  y = wx +b

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

X = torch.from_numpy(income.Education.values.reshape(-1, 1)).type(torch.FloatTensor)
Y = torch.from_numpy(income.Income.values)
learning_rate = 0.0001
for epoch in range(10000):
    for x, y in zip(X, Y):
        y_pred = torch.matmul(x, w) + b
        loss = (y - y_pred).pow(2).sum()

        if w.grad is not None:
            w.grad.data.zero_()
        if b.grad is not None:
            b.grad.data.zero_()

        loss.backward()

        with torch.no_grad():
            w.data -= w.grad.data * learning_rate
            b.data -= b.grad.data * learning_rate

    if epoch % 100 == 0:
        y_ = torch.matmul(X, w) + b
        l = (Y - y_).pow(2).mean()
        print(w.data, b.data, l.data)


model = LinearRegression()

model.fit(X,Y)

print(model.coef_)
print(model.intercept_)


plt.scatter(income.Education, income.Income)
plt.plot(X.numpy(), (torch.matmul(X, w) + b).data.numpy(), c='g')
plt.xlabel('Education')
plt.ylabel('Income')
plt.show()




