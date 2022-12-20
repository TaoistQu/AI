#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/20 20:54
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : pca.py
# @Software: PyCharm
# description: Principal Components Analysis 是一种线性的方式

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

X,y = datasets.load_iris(return_X_y=True)

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=102)
pca = PCA(n_components=2,whiten=True)
pca.fit(X)
X_pca = pca.transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_pca,y,test_size=0.2,random_state=102)

#print(X[:5],X_pca[:5],X_pca.mean(axis=0),X_pca.std(axis=0))
print(X_pca.mean(axis=0))
print(X_pca.shape)
print(X_pca.std(axis=0))

model = LogisticRegression()
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
print(score)
