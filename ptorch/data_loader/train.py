# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :train.py
# @Time      :2023/4/2 下午9:56
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:
import pandas as pd
import torch

from ptorch.data_loader import data_model
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
#/media/ql/hdd/code/AI/ptorch/data/HR.csv
#/media/ql/hdd/code/AI/ptorch/data_loader/train.py
data = pd.read_csv('/media/ql/hdd/code/AI/ptorch/data/HR.csv')
data = data.join(pd.get_dummies(data.part)).join(pd.get_dummies(data.salary))
data.drop(columns=['part', 'salary'], inplace=True)

X_data = data[[c for c in data.columns if c != 'left']].values
Y_data = data.left.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

smote = SMOTE()

X_smoted, Y_smoted = smote.fit_resample(X_scaled, Y_data)
X = torch.from_numpy(X_smoted).type(torch.FloatTensor)
Y = torch.from_numpy(Y_smoted.reshape(-1, 1)).type(torch.FloatTensor)

lr = 0.001
batch_size = 64
epochs = 100

data_model.data_loader_train(X, Y, lr, batch_size, epochs)