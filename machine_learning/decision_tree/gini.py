#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 20:11
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : gini.py
# @Software: PyCharm
# description:gini系数作为分裂条件

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree
import os

y = np.array(list('NYYYYYNYYN'))
X = pd.DataFrame({'日志密度':list('sslmlmmlms'),
                  '好友密度':list('slmmmlsmss'),
                  '真实头像':list('NYYYYNYYYY')})

X['日志密度'] = X['日志密度'].map({'s':0,'m':1,'l':2})
X['好友密度'] = X['好友密度'].map({'s':0,'m':1,'l':2})
X['真实头像'] = X['真实头像'].map({'N':0,'Y':1})


model = DecisionTreeClassifier(criterion='gini')

model.fit(X,y)
path = os.path.abspath('../image')

dot_data = tree.export_graphviz(model, out_file=os.path.join(path,'./gini'),
                                feature_names=X.columns,
                                class_names=np.unique(y),
                                filled=True,
                                rounded=True,
                                fontname='kaiTi')
graph = graphviz.Source.from_file(os.path.join(path,'.\gini'))
graph.render(os.path.join(path,'.\gini'),format='png')# 保存到文件中

X['真实用户'] = y
columns = ['日志密度','好友密度','真实头像']
lower_gini = 1
condition = {}
for col in columns:
    x = X[col].unique()
    x.sort()

    for i in range(len(x) - 1):
        split = x[i:i+2].mean()
        cond = X[col] <= split
        p = cond.value_counts() / cond.size
        indexes = p.index
        gini = 0
        for index in indexes:
            user = X[cond == index]['真实用户']
            p_user = user.value_counts() / user.size
            gini += (p_user *(1-p_user)).sum() * p[index]
        print(col,split,gini)
        if gini < lower_gini:
            condition.clear()
            lower_gini = gini
            condition[col] = split
print('最佳裂分条件是：',condition)

