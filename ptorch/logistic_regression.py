# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :logistic_regression.py
# @Time      :2023/3/27 下午9:10
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

data = pd.read_csv('data/credit-a.csv', header=None)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

Y.replace(-1, 0, inplace=True)

X = torch.from_numpy(X.values).type(torch.FloatTensor)
Y = torch.from_numpy(Y.values.reshape(-1, 1)).type(torch.FloatTensor)
print(X.shape)

model = nn.Sequential(
    nn.Linear(15, 512),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

batch_size = 32
steps = 653 //32

for epoch in range(10000):
    for batch in range(steps):
        start = batch * batch_size
        end = start + batch_size

        x = X[start:end]
        y = Y[start:end]

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


dict = model.state_dict()
for thresh in range(10):
    mean = ((model(X).data.numpy() >= 0.1*thresh) == Y.numpy()).mean()
    print(thresh)
    print(mean)