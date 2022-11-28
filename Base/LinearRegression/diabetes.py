#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/28 23:21
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : diabetes.py
# @Software: PyCharm

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

diabetes = load_diabetes()
data = diabetes['data']
target = diabetes['target']

X_train,X_test,y_train,y_test = train_test_split(data,target)
linear = LinearRegression()

linear.fit(X_train,y_train)
print(linear.coef_)
print(linear.score(X_test,y_test))

