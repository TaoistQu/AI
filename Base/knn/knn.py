#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/26 18:04
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : knn.py
# @Software: PyCharm

from IPython.core.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

X,y = datasets.load_iris(return_X_y=True)# 鸢尾花，分三类，鸢尾花花萼和花瓣长宽
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pre = knn.predict(X_test)
acc = (y_test == knn.predict(X_test)).mean()
print(knn.score(X_test,y_test))
display(acc)

