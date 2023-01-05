#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 21:23
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : entropy.py
# @Software: PyCharm
# description: 使用交叉熵作为分裂条件,按照日志密度进行划分

import numpy as np
import pandas as pd

y = np.array(list('NYYYYYNYYN'))
X = pd.DataFrame({'日志密度':list('sslmlmmlms'),
                  '好友密度':list('slmmmlsmss'),
                  '真实头像':list('NYYYYNYYYY')})

X['日志密度'] = X['日志密度'].map({'s': 0, 'm': 1, 'l' : 2})
X['好友密度'] = X['好友密度'].map({'s': 0, 'm': 1, 'l': 2})
X['真实头像'] = X['真实头像'].map({'N': 0, 'Y': 1})
#去重
x = X['日志密度'].unique()
x.sort()
X['真实用户'] = y

for i in range(len(x) - 1):
    split = x[i:i+2].mean()
    cond = X['日志密度'] <= split
    p = cond.value_counts() / cond.size
    indexes = p.index

    entropy = 0
    for index in indexes:
        user = X[cond == index]['真实用户']
        p_user = user.value_counts() / user.size
        entropy += (p_user *np.log2(1/p_user)).sum() * p[index]

    print(split, entropy)



