#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/1 12:39
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : gauss.py
# @Software: PyCharm
# description:高斯滤波

import numpy as np
import cv2
import os

# 计算(0,0)坐标点, 对应的值
sigma = 1 / (2 * np.pi * 1.5**2)
print(sigma)
# 计算(-1, 1)坐标点对应的值
sigma = 1 / (2 * np.pi * 1.5**2)* np.exp(-(2/(2*1.5**2)))
print(sigma)

path = os.path.abspath('../images')

img = cv2.imread(os.path.join(path,'./lena_color.png'))

new_img = cv2.GaussianBlur(img, (3,3), sigmaX=100)
cv2.imshow('img', np.hstack((img,new_img)))

cv2.waitKey(0)
cv2.destroyAllWindows()