#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/20 22:54
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : cov.py
# @Software: PyCharm
# description: 计算协方差

import numpy as np
A = np.random.randint(0,10,size=(3,3))
C = np.random.randint(0,10,size=(3,4))
cov = np.cov(A,rowvar=True)

B = (A - A.mean(axis=1).reshape(-1,1))
print(C)
print(C.mean(axis=0))
#print(B)
scatter = B.dot(B.T)

#print(cov)
#print(scatter / (3-1))
