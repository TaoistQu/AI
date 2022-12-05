#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/6 0:37
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : max_min.py
# @Software: PyCharm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

X_1 = np.random.randint(1,10,size=10)
X_2 = np.random.randint(100,300,size=10)

X = np.c_[X_1,X_2]
print('归一化前的数据：')
print(X)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
#X = (X - X_min) / (X_max - X_min)

print(X)

scaler = MinMaxScaler()
y = scaler.fit_transform(X)
print(y)
