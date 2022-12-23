#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/22 20:39
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : iris.py
# @Software: PyCharm
# description:

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import os

iris = load_iris()
clf = DecisionTreeClassifier()
clf_iris = clf.fit(iris.data, iris.target)

# 将决策树的结构数据导出
path  = os.path.abspath('../image')
dot_data = tree.export_graphviz(clf, out_file =os.path.join(path,'.\iris') )
graph = graphviz.Source.from_file(os.path.join(path,'.\iris'))
graph.render(os.path.join(path,'.\iris'),format='pdf')# 保存到文件中
