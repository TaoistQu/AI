#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/22 15:54
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : lad.py
# @Software: PyCharm
# description:
import numpy as np
from scipy import linalg
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
np.set_printoptions(suppress=True)
X,y = load_iris(return_X_y=True)

lda = LinearDiscriminantAnalysis(solver='eigen',n_components=2)

X_lda = lda.fit_transform(X,y)

## 1、总的散度矩阵
# 协方差，计算的列的,Scatter _ total

St = np.cov(X.T,rowvar=True,bias=1)
# 2、类内散度矩阵，分3类：0、1、2
# Sw within类内

Sw = np.full(shape=(4,4),fill_value=0,dtype=np.float64)
for i in range(3):
    Sw += np.cov(X[y==i],rowvar=False,bias=1)
Sw /= 3
# 3、计算类间的散度矩阵

Sb = St - Sw
#4、计算特征值和特征向量
eigen,ev = linalg.eigh(Sb,Sw)
np.argsort(eigen)
n_components = 2
ev = ev[:,[3,2,1,0]][:,:n_components]
#进行矩阵乘法（投影）
lda_result = X.dot(ev)[:5]
print(lda_result[:5])
print(X_lda[:5])