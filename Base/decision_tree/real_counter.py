#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/22 19:54
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : real_counter.py
# @Software: PyCharm
# description:

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'KaiTi'

y = np.array(list('NYYYYYNYYN'))

X = pd.DataFrame({'日志密度':list('sslmlmmlms'),
                  '好友密度':list('slmmmlsmss'),
                  '真实头像':list('NYYYYNYYYY')})
#特征工程，对数据进行清洗
X['日志密度'] = X['日志密度'].map({'s':0,'m':1,'l':2})
X['好友密度'] = X['好友密度'].map({'s':0,'m':1,'l':2})
X['真实头像'] = X['真实头像'].map({'N':0,'Y':1})

model = DecisionTreeClassifier()
model.fit(X,y)
y_ = model.predict(X)
print(y)
print(y_)

plt.figure(figsize=(12,16))
tree.plot_tree(model,filled=True,feature_names=['日志密度','好友密度','真实头像'])
plt.show()