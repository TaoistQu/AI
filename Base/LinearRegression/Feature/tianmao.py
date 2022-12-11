#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/11 16:44
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : tianmao.py
# @Software: PyCharm

import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.size'] = 18
plt.figure(figsize=(9,6))
np.set_printoptions(suppress=True)

X = np.arange(2009,2020)
y = np.array([0.5,9.36,52,191,350,571,912,1207,1682,2135,2684])
#可视化原始数据
#plt.bar(X, y, width = 0.5, color = 'green')
#plt.plot(X, y, color = 'red')
#将数据升成二维
X = X.reshape(-1,1)

#对数据进行处理
#X -= np.int32(X.mean())
X -= 2008
poly = PolynomialFeatures(degree=3,include_bias=False)

X = poly.fit_transform(X)
#最大最小值归一化
mm = MinMaxScaler()
X_mm = mm.fit_transform(X)
#0均值归一化
standard = StandardScaler()
X_s = standard.fit_transform(X)

model_mm = SGDRegressor(fit_intercept=True,max_iter=50000,eta0=0.5)
model_s = SGDRegressor(fit_intercept=True,max_iter=50000,eta0=0.5)

model_mm.fit(X_mm,y)
model_s.fit(X_s,y)

X_test = np.linspace(1,12).reshape(-1,1)
X_test = poly.fit_transform(X_test)

X_test_s = standard.transform(X_test)
X_test_mm = mm.transform(X_test)

y_mm_ = model_mm.predict(X_test_mm)
y_s_ = model_s.predict(X_test_s)

plt.scatter(np.arange(2009,2020),y,color = 'red')
plt.plot(np.linspace(2009,2020),y_s_,color = 'green')
plt.plot(np.linspace(2009,2020),y_mm_,color = 'blue')

plt.show()