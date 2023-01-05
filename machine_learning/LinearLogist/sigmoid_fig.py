#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/11 19:44
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : sigmoid_fig.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
np.set_printoptions(suppress=True)
def sigmoid(x):
    return 1/(1+np.exp(-x))

x= np.linspace(-5,5,100)
y = sigmoid(x)
plt.plot(x,y,color='green')
plt.title('sigmoid_fig')
plt.xlabel('x')
plt.ylabel('sigmoid')
plt.show()

X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_ = model.predict(X_test)
print('真实类别：',y_test)
print('预测类别：',y_)
score = model.score(X_test,y_test)
print(score)
proba= model.predict_proba(X_test)
print(proba)