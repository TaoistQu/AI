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
import joblib
import numpy as np

X,y = datasets.load_iris(return_X_y=True)# 鸢尾花，分三类，鸢尾花花萼和花瓣长宽
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)
display(X_train)
knn.fit(X_train,y_train)
y_ = knn.predict(X_test)
acc = (y_test == knn.predict(X_test)).mean()
print(knn.score(X_test,y_test))
display(acc)
#模型保存
joblib.dump(knn,'./model')
model = joblib.load('./model')
print(type(model))
y_ = model.predict(X_test)
display(y_)
score = model.score(X_test,y_test)
display(score)

X_new = np.array([[5.4,3.2,0.8,2.3],
                  [1.2,3.4,5.6,6.8]])
y_new_ = model.predict(X_new)
display(y_new_)


