#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/15 23:33
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : sofmax.py
# @Software: PyCharm
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True)

X,y = datasets.load_iris(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15)

model = LogisticRegression(multi_class='multinomial')
model.fit(X_train,y_train)

y_ = model.predict(X_test)
print(y_,y_test)

proba_ = model.predict_proba(X_test)

w_ = model.coef_
b_ = model.intercept_

z = X_test.dot(w_.T) + b_
def softmax(z):
    return np.exp(z) / (np.exp(z).sum(axis=1)).reshape(-1,1)
pro = softmax(z)

print(pro.argmax(axis=1))

