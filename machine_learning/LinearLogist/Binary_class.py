#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/14 16:21
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : Binary_class.py
# @Software: PyCharm

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

X,y = datasets.load_iris(return_X_y=True)
cond = y != 2
X = X[cond]
y = y[cond]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15)

model = LogisticRegression()
model.fit(X_train,y_train)

y_ = model.predict(X_test)
print('预测的y：',y_)
print('真实的y：',y_test)

proba_ = model.predict_proba(X_test)
print(proba_)

print(proba_.argmax(axis=1))

w_ = model.coef_
b_ = model.intercept_

print(w_,b_)

z = X_test.dot(w_.T) + b_

def sigmoid(z):
    return 1/(1+np.exp(-z))

p = sigmoid(z)

pro = np.concatenate([1-p,p],axis=1)
print(pro)

print(pro.argmax(axis=1))

