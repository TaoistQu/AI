#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 21:13
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : hyperparameter.py
# @Software: PyCharm
# description: 超参数求解

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt



iris = datasets.load_iris()
X = iris['data']
y = iris['target']
name = iris['target_names']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=256)

depth = np.arange(1,16)
err = []
for d in depth:
    model = DecisionTreeClassifier(criterion='entropy',max_depth=d)
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    err.append(1-score)

plt.figure(figsize=(9,6))
plt.rcParams['font.family'] = 'STKaiti'
plt.plot(depth,err,'ro-')
plt.xlabel('决策树深度',fontsize=16)
plt.ylabel('错误率', fontsize=18)
plt.grid()
plt.show()

