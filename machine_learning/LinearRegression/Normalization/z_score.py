#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/6 0:54
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : z_score.py
# @Software: PyCharm
import numpy as np
from sklearn.preprocessing import StandardScaler
X_1 = np.random.randint(1,10,size=10)
X_2 = np.random.randint(100,300,size=10)

X = np.c_[X_1,X_2]
print(X)
x_ = (X - X.mean(axis=0)) / X.std(axis=0)
print(x_)

standard_scaler = StandardScaler()
standard_scaler.fit(X)
x_std = standard_scaler.transform(X)
print(x_std)
