# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :linear_regression_portch.py
# @Time      :2023/3/27 下午8:41
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:
from torch import nn
import torch
import pandas as pd

income = pd.read_csv('data/Income1.csv')
X = torch.from_numpy(income.Education.values.reshape(-1, 1)).type(torch.FloatTensor)
Y = torch.from_numpy(income.Income.values).type(torch.FloatTensor)

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()

model.parameters()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    for x, y in zip(X, Y):
        y_pred = model(x)
        loss = loss_fn(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



print(model.weight)