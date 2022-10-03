#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/3 14:20
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : softmax.py
# @Software: PyCharm

import numpy as np

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y