#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/11 17:30
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : china_life.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.linear_model import ElasticNet

data = pd.read_excel('./中国人寿.xlsx')

print(data.shape)

fig,axes = plt.subplots(2,2,figsize=(12,9))

sns.kdeplot(data['charges'],shade=True,hue = data['sex'],ax = axes[0,0])
# 地区对保费影响
sns.kdeplot(data['charges'],shade = True,hue = data['region'],ax = axes[0,1])

# 吸烟对保费影响
sns.kdeplot(data['charges'],shade = True,hue = data['smoker'],ax = axes[1,0])

# 孩子数量对保费影响
sns.kdeplot(data['charges'],shade = True,hue = data['children'],palette='Set1',ax = axes[1,1])
plt.show()
#去除影响不大的数据
data = data.drop(['region','sex'],axis=1)
#print(data.head())

def convert(df,bmi):
    df['bmi'] = 'fat' if df['bmi'] >= bmi else 'standard'
    return df
data = data.apply(convert,axis= 1 , args = (30,))

#### 字符串独热编码数据转换
#特征提取

data = pd.get_dummies(data)
#### 数据提取
#print(data.head())
X = data.drop('charges',axis=1)
y = data['charges']
print(X.head())
#print(y.head())

#特征升维
poly_int = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
X_int = poly_int.fit_transform(X)

poly_no = PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
X_no = poly_no.fit_transform(X)


print(X_int[1])
print(X_no[1])
#拆分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X_int,y,test_size=0.2,random_state=1024)

#建模训练
model_1 = LinearRegression()
model_1.fit(X_train,y_train)
#评估R_2评估
print('测试数据得分：',model_1.score(X_train,y_train))
print('预测数据得分：',model_1.score(X_test,y_test))

# 均方误差
print('训练数据均方误差：',np.sqrt(mean_squared_error(y_train, model_1.predict(X_train))))
print('测试数据均方误差：',np.sqrt(mean_squared_error(y_test, model_1.predict(X_test))))

# 均方误差求对数
print('训练数据对数误差：',np.sqrt(mean_squared_log_error(y_train,model_1.predict(X_train))))
print('测试数据对数误差：',np.sqrt(mean_squared_log_error(y_test,model_1.predict(X_test))))

model_2 = ElasticNet(alpha=0.01,l1_ratio=0.1,max_iter=5000)
model_2.fit(X_train,y_train)

print('测试数据得分：',model_2.score(X_train,y_train))
print('预测数据得分：',model_2.score(X_test,y_test))

print('训练数据均方误差为：',np.sqrt(mean_squared_error(y_train,model_2.predict(X_train))))
print('测试数据均方误差为：',np.sqrt(mean_squared_error(y_test,model_2.predict(X_test))))

print('训练数据对数误差为：',np.sqrt(mean_squared_log_error(y_train,model_2.predict(X_train))))
print('测试数据对数误差为：',np.sqrt(mean_squared_log_error(y_test,model_2.predict(X_test))))
