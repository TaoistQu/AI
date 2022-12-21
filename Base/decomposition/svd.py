#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/21 16:36
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : svd.py
# @Software: PyCharm
# description:奇异值分解

import numpy as np
from IPython.core.display import display

A = np.random.randint(0,10,size=(5,3))
U,S,Vt = np.linalg.svd(A,full_matrices=False)
display(U,S,Vt)
A_ = U.dot(np.diag(S)).dot(Vt)
print(A_)
print(A)

