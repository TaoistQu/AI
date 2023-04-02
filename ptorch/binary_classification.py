# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :binary_classification.py
# @Time      :2023/3/28 下午9:14
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import torch
from torch import nn

data = pd.read_csv('data/HR.csv')

data = data.join(pd.get_dummies(data.part)).join(pd.get_dummies(data.salary))
data.drop(columns=['part', 'salary'], inplace=True)




class HRModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin_1 = nn.Linear(20, 64)
        self.lin_2 = nn.Linear(64, 64)
        self.lin_3 = nn.Linear(64, 1)
        self.activate = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.lin_1(input)
        x = self.activate(x)
        x = self.lin_2(x)
        x = self.activate(x)
        x = self.lin_3(x)
        x = self.sigmoid(x)
        return x


lr = 0.001

def get_model():
    model = HRModel()
    return model, torch.optim.Adam(model.parameters(), lr=lr)

batch_size = 64
steps = len(data) // batch_size

epochs =100
def train(X, Y, model, opt, loss_fn):
    for epoch in range(epochs):
        for i in range(steps):
            start = i * batch_size
            end = start + batch_size

            x = X[start:end]
            y = Y[start:end]

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print('epoch:',epoch,'     ', 'loss:', loss_fn(model(X), Y))

def accuracy(model, X ,Y):
    print(((model(X).data.numpy() >= 0.3) == Y.numpy()).mean())
    print(np.unique(model(X).data.numpy()))

'''
for thresh in range(10):
    mean = ((model(X).data.numpy() >= 0.1*thresh) == Y.numpy()).mean()
    print(thresh)
    print(mean)
'''

loss_fn = nn.BCELoss()

model, opt = get_model()


#处理数据不平衡

X_data =data[[c for c in data.columns if c!= 'left']].values
Y_data = data.left.values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_data)


smote = SMOTE()
X, Y = smote.fit_resample(X_scaled, Y_data)
print(X.shape)
X = torch.from_numpy(X).type(torch.FloatTensor)
Y = torch.from_numpy(Y.reshape(-1, 1)).type(torch.FloatTensor)

train(X, Y, model, opt, loss_fn)
accuracy(model, X, Y)