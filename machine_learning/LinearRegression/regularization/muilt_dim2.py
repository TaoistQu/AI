#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/11 16:16
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : muilt_dim2.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

X = np.linspace(-1,11,num=100)
y = (X-5)**2 + 3*X -12 + np.random.randn(100)
X = X.reshape(-1,1)
plt.scatter(X,y)
print('未升维前X：',X)
X_test = np.linspace(-2,12,num=200).reshape(-1,1)

poly = PolynomialFeatures()
poly.fit(X,y)
X = poly.transform(X)
print('升维后的X: ',X)
model = SGDRegressor(penalty='l2',eta0=0.001,max_iter=10000)
model.fit(X,y)
X_test = poly.transform(X_test)
y_test = model.predict(X_test)
plt.plot(X_test[:,1],y_test,color='green')

#进行归一化
s = StandardScaler()
s.fit(X)
X = s.transform(X)
X_test_norl = s.transform(X_test)
model2 = SGDRegressor(penalty='l2',eta0=0.001,max_iter=10000)
model2.fit(X,y)
y_test = model2.predict(X_test_norl)
plt.plot(X_test[:,1],y_test,color='red')
plt.show()

