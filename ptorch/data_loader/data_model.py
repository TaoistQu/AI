# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :data_model.py
# @Time      :2023/4/2 下午8:51
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:dataset 返回一批数据

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class HRModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin1 = nn.Linear(20, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)
        self.activate = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activate(x)
        x = self.lin2(x)
        x = self.activate(x)
        x = self.lin3(x)
        x = self.sigmoid(x)

        return x

def get_model(lr):
    model = HRModel()
    return model,torch.optim.Adam(model.parameters(), lr=lr)

def accuracy(model, X, Y):
    print(((model(X).data.numpy() >= 0.3) == Y.numpy()).mean())
    print(np.unique(model(X).data.numpy()))

def train(X, Y, lr, batch_size, epochs):
    steps = len(X) // batch_size
    HRDataset = TensorDataset(X, Y)
    model, opt = get_model(lr)
    for epoch in range(epochs):
        for i in range(steps):
            start = i * batch_size
            end = start + batch_size
            x, y = HRDataset[start: end]

            y_pred = model(x)
            loss = nn.BCELoss()(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('epoch:', epoch, '     ', 'loss: ', nn.BCELoss()(model(X), Y))
    accuracy(model, X, Y)


def data_loader_train(X, Y, lr, batch_size, epochs):
    steps = len(X) // batch_size
    HR_ds = TensorDataset(X, Y)
    HR_dl = DataLoader(HR_ds, batch_size=batch_size, drop_last=True)
    model, opt = get_model(lr)
    for epoch in range(epochs):
        for x, y in HR_dl:
            y_pred = model(x)
            loss = nn.BCELoss()(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('epoch:', epoch, '     ', 'loss: ', nn.BCELoss()(model(X), Y))
    accuracy(model, X, Y)





