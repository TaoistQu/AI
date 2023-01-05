#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/21 16:02
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : pca_eig.py
# @Software: PyCharm
# description: 基于特征值和特征向量实现的主成分分析
#*1、去平均值(即去中心化)，即每一位特征减去各自的平均值
#2、计算协方差矩阵
# 3、用特征值分解方法求协方差矩阵的特征值与特征向量
# 4、对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为行向量组成特征向量矩阵ev
# 5.将数据转换到k个特征向量构建的新空间中，即X_pca= $X \cdot

from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

X,y = datasets.load_iris(return_X_y=True)

pca = PCA(n_components=0.8,whiten=True)
X_pca = pca.fit_transform(X)
print(X_pca[:5])

#去中心化
B = X - X.mean(axis=0) #按列操作
#计算协方差
V = np.cov(B,rowvar=False,bias=True)
#特征值和特征向量计算
w,v = np.linalg.eig(V)
#print(w)
#print(v)
# 符号翻转，绝对值最大的，如果是负数，才翻转
max_abs_cols = np.argmax(np.abs(v),axis=0)
signs = np.sign(v[max_abs_cols,[0,1,2,3]])
v *= signs
#特征值筛选
cond = (w/w.sum()).cumsum() >= 0.8
index = cond.argmax()
v_ = v[:,:index+1]

#矩阵运算得到pca矩阵
pca_result = B.dot(v_)
#标准化
pca_result = (pca_result - pca_result.mean(axis=0)) / pca_result.std(axis=0,ddof=1)
print(pca_result[:5])

