#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/28 23:33
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : boston.py
# @Software: PyCharm


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

boston = datasets.load_boston()
data = boston['data']
target = boston['target']


X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=50)

mode = LinearRegression(fit_intercept=True)
mode.fit(X_train,y_train)
y_ = mode.predict(X_train)
print(y_[:5])
print(y_test[:5])
print(mode.score(X_train,y_train))





