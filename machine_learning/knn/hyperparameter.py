#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/26 21:38
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : hyperparameter.py
# @Software: PyCharm
from IPython.core.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X,y= datasets.load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=11)
knn = KNeighborsClassifier()
scores = []
for k in range(2,50):
    knn.set_params( n_neighbors = k) #  n_neighbors = k
    knn.fit(X_train,y_train)
    score = knn.score(X_test,y_test)
    scores.append(score)

plt.plot(np.arange(2,50),scores,'*r-')
plt.show()

knn.set_params(n_neighbors=5,weights='distance')
knn.fit(X_train,y_train)
display(knn.score(X_test,y_test))

