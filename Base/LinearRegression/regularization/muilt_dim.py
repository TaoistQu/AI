#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/11 15:55
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : muilt_dim.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.set_printoptions(suppress=True)
X = np.linspace(-1,11,num = 100)
y = (X-5)**2 + 3 * X -12 + np.random.randn(100)
X = X.reshape(-1,1)
plt.scatter(X,y)

X_test = np.linspace(-2,12,num=200).reshape(-1,1)

#不进行升维，直接用普通线性回归
model_1 = LinearRegression()
model_1.fit(X,y)
y_test = model_1.predict(X_test)

plt.plot(X_test,y_test,color='red')



#进行多项式升维 + 普通线性回归预测
print('升维前X:',X)
X = np.concatenate([X,X**2],axis=1)
print('升维后X:',X)
model_2 = LinearRegression()
model_2.fit(X,y)
X_test = np.concatenate([X_test,X_test**2],axis=1)
y_test = model_2.predict(X_test)
plt.plot(X_test[:,0],y_test,color='green')
plt.show()
