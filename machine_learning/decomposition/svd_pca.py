#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/21 17:20
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : svd_pca.py
# @Software: PyCharm
# description:使用奇异值分解计算pca

from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
X,y = datasets.load_iris(return_X_y=True)

pca = PCA(n_components=0.95,whiten=True)
X_pca = pca.fit_transform(X)
print(X_pca[:5])

n_components = 2
#去中心化
B = X - X.mean(axis=0)

U,S,Vt = np.linalg.svd(B,full_matrices=False)
# 3、符号翻转
# 符号翻转，绝对值最大的，如果是负数，才翻转
max_abs_cols = np.argmax(np.abs(U),axis=0)
signs = np.sign(U[max_abs_cols,[0,1,2,3]])
U *= signs

# 4、降维特征筛选
U = U[:,:n_components]
#归一化
U = (U - U.mean(axis=0)) / (U.std(axis=0,ddof=1))
print(U[:5])
