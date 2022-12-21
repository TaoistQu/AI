#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/21 15:51
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : eigenvalue.py
# @Software: PyCharm
# description: 特征值和特征向量

import numpy as np

A = np.random.randint(0,10,size=(3,3))

w,v = np.linalg.eig(A)

print(w)
print(v)
#用矩阵乘以特征向量 == 特征值 乘以 特征向量

print(A.dot(v[:,1]))
print(w[1] * v[:,1])
print(v[:,0])

w_ = np.diag(w)
print(w_)
A_ = v.dot(w_).dot(np.linalg.inv(v))
print(A_)
print(A)



