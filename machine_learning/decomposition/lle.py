#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/22 18:07
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : lle.py
# @Software: PyCharm
# description:

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt

X,t = datasets.make_swiss_roll(n_samples=1500, noise=0.05, random_state=100)

fig = plt.figure(figsize=(12, 9))

axes3D = fig.add_subplot(projection = '3d')
axes3D.scatter(X[:, 0], X[:, 1], X[:, 2], c=t)
axes3D.view_init(7, 80)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0],X_pca[:, 1],c=t)

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_lle = lle.fit_transform(X)
plt.subplot(1, 2, 2)
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=t)
plt.show()
