#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 13:30
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : mult_entropy.py
# @Software: PyCharm
# description:多个特征进行最佳裂分条件
import numpy as np
import pandas as pd

y = np.array(list('NYYYYYNYYN'))
X = pd.DataFrame({'日志密度':list('sslmlmmlms'),
                  '好友密度':list('slmmmlsmss'),
                  '真实头像':list('NYYYYNYYYY')})

X['日志密度'] = X['日志密度'].map({'s':0,'m':1,'l':2})
X['好友密度'] = X['好友密度'].map({'s':0,'m':1,'l':2})
X['真实头像'] = X['真实头像'].map({'N':0,'Y':1})

X['真实用户'] = y

columns = ['日志密度','好友密度','真实头像']

lower_entropy = 1
condition ={}

for col in columns:
    x = X[col].unique()
    x.sort()

    for i in range(len(x) -1):
        split = x[i:i+2].mean()
        cond = X[col] <= split
        p = cond.value_counts() / cond.size
        indexes = p.index
        entropy = 0
        for index in indexes:
            user = X[cond == index]['真实用户']
            p_user = user.value_counts() / user.size
            entropy += (p_user*np.log2(1/p_user)).sum()*p[index]

        print(col,split,entropy)
        if entropy < lower_entropy:
            condition.clear()
            lower_entropy = entropy
            condition[col] = split
print('最佳分裂条件是：',condition)



