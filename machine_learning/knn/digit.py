#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/26 22:04
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : digit.py
# @Software: PyCharm
import numpy as np
from IPython.core.display import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = np.load('..\data\digit.npy')
print(data.shape)
index = np.random.randint(0,5000,size=1)[0]
print(index)
plt.figure(figsize=(2,2))
plt.imshow(data[index])
plt.show()

y = np.array([0,1,2,3,4,5,6,7,8,9]*500)
y = np.sort(y)
data = data.reshape(5000,-1)
X_train,X_test,y_train,y_test = train_test_split(data,y,test_size=0.1)
print(X_train.shape,X_test.shape)
print(X_train.ndim)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)

y_ = knn.predict(X_test)
display(y_test[:20],y_[:20])
score = knn.score(X_test,y_test)
print(score)

plt.figure(figsize=(2*5,3*10))
for i in range(50):
    plt.subplot(10,5,i+1)
    plt.imshow(X_test[i].reshape(28,28))
    plt.axis('off')
    plt.title('True:%d\nPredict:%d'%(y_test[i],y_[i]))
plt.show()



