#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/21 20:40
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : mnist.py
# @Software: PyCharm
# description: 使用手写数字数据集测试

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
data = pd.read_csv('D:\MyCode\AI\Base\data\digits.csv')

y = data['label'].values
X = data.iloc[:,1:]

image = X.iloc[4].values.reshape(28,28)
plt.figure(figsize=(2,2))
plt.imshow(image,cmap='gray')
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = LogisticRegression(solver='lbfgs',max_iter=1000)
model.fit(X_train,y_train)
y_ = model.predict(X_test)
print(y_[:20])
print(y_test[:20])
print(model.score(X_test,y_test))

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

X_train_pca,X_test_pca,y_train_pca,y_test_pca = train_test_split(X_pca,y,test_size=0.2)
model_pca = LogisticRegression()
model_pca.fit(X_train_pca,y_train_pca)
y_pca_ = model_pca.predict(X_test_pca)
print(y_pca_[:20])
print(y_test_pca[:20])
print(model_pca.score(X_test_pca,y_test_pca))