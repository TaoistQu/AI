#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/31 14:51
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : warpAffine.py
# @Software: PyCharm
# 图像仿射变化
# 仿射变换是图像旋转, 缩放, 平移的总称.
# 具体的做法是通过一个矩阵和和原图片坐标进行计算,
# 得到新的坐标, 完成变换. 所以关键就是这个矩阵.

import cv2
import numpy as np
import os

path = os.path.abspath('../images')

dog = cv2.imread(os.path.join(path,'./dog.jpeg'))

h, w, ch = dog.shape
M = np.float32([[1,0,100],[0,1,0]])

# 平移操作
# 注意opencv中是先宽度, 后高度.

new_dog = cv2.warpAffine(dog, M,dsize=(w,h))

cv2.imshow('dog',dog)
cv2.imshow('new_dog',new_dog)

cv2.waitKey(0)
cv2.destroyAllWindows()


