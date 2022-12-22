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

iris = load_iris()
clf = DecisionTreeClassifier()
clf_iris = clf.fit(iris.data, iris.target)

# 将决策树的结构数据导出
dot_data = tree.export_graphviz(clf, out_file = 'D:\MyCode\AI\Base\image\iris')
graph = graphviz.Source.from_file('D:\MyCode\AI\Base\image\iris')
graph.render('D:\MyCode\AI\Base\image\iris',format='pdf')# 保存到文件中
