# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :torchvision_loader.py
# @Time      :2023/4/17 下午9:58
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch import nn
#%%
import torch.nn.functional as F

transformation = transforms.Compose([transforms.ToTensor(), ])

train_ds = datasets.MNIST(root='../data/', train=True, transform=transformation)
test_ds = datasets.MNIST('../data/', train=False, transform=transformation)



#%%
# 变成data loader
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 写init按照模型结果的顺序去写.
        self.conv1 = nn.Conv2d(1, 32, 3)  # in: 64, 1, 28, 28 -> out: 64, 32, 26, 26
        self.pool = nn.MaxPool2d(2, 2)  # out: 64, 32, 13, 13
        self.conv2 = nn.Conv2d(32, 64, 3)  # out: 64, 64, 11, 11 -> maxpool : 64, 64, 5, 5

        self.linear_1 = nn.Linear(64 * 5 * 5, 256)
        self.linear_2 = nn.Linear(256, 10)
        self.flatten = nn.Flatten()

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


def train_one_epoch(model, train_loader,test_loader):
    correct = 0
    total = 0
    running_loss = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #计算训练的损失
        with torch.no_grad():
            y_ = torch.argmax(y_pred, dim=1)
            correct += (y_ == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device),y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_ = torch.argmax(y_pred, dim=1)
            test_correct += (y_ == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    test_epoch_loss = test_running_loss / len(test_loader.dataset)
    test_epoch_acc = test_correct / test_total

    print('epoch: ', epoch,
          'loss: ', round(epoch_loss, 3),
          'accuracy: ', round(epoch_acc, 3),
          'test_loss: ', round(test_epoch_loss, 3),
          'test_accuracy: ', round(test_epoch_acc, 3))
    return epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc




model = Model()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
#%%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
train_loss = []
train_acc = []
test_loss = []
test_acc = []
epochs = 30
# pytorch默认是在cpu上跑的, cpu比较慢
for epoch in range(epochs):
    epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(model, train_dl, test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(test_epoch_loss)
    test_acc.append(test_epoch_loss)


plt.plot(train_loss, label='train_loss')
plt.plot(train_acc, label='train_acc')
plt.plot(test_loss, label='test_loss')
plt.plot(test_acc,   label='test_acc')
plt.legend()
plt.show()