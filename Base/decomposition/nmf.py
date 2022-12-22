#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/22 16:46
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : nmf.py
# @Software: PyCharm
# description:
from sklearn import datasets
from sklearn.decomposition import NMF

X,y = datasets.load_iris(return_X_y=True)

nmf = NMF(n_components=2,max_iter=1000)

W = nmf.fit_transform(X)
print(W[:5])

H = nmf.components_

print(H)

X_ = W.dot(H)

print(X[:5])
print(X_[:5])