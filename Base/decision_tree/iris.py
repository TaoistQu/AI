#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 20:44
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : iris.py
# @Software: PyCharm
# description:

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
import os

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
name = iris['target_names']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=256)
model = DecisionTreeClassifier(max_depth=None,criterion='entropy')
model.fit(X_train,y_train)
y_ = model.predict(X_test)

print('真实类别是：',y_test)
print('算法预测是：',y_)
print('准确率是：',model.score(X_test,y_test))
path = os.path.abspath('../image')
dot_data = tree.export_graphviz(model,feature_names=iris['feature_names'],
                                out_file=os.path.join(path,'./iris'),
                                class_names=name,
                                filled=True,
                                rounded=True,
                                fontname='kaiTi',)

graph = graphviz.Source.from_file(os.path.join(path,'./iris'))
graph.render(os.path.join(path,'./iris'),format='png')



