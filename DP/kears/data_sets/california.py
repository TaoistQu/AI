# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :california.py
# @Time      :2023/3/5 下午9:58
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data():

    housing = fetch_california_housing()

    data = housing.data
    target = housing.target

    x_train_all, x_test, y_train_all, y_test = train_test_split(data, target, random_state=7)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all,random_state=10)

    #正态标准化
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_valid_scaled, x_test_scaled, y_train, y_valid, y_test, housing.feature_names

