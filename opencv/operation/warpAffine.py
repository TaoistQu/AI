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
'''
getRotationMatrix2D(center, angle, scale)

center 中心点 , 以图片的哪个点作为旋转时的中心点.
angle 角度: 旋转的角度, 按照逆时针旋转.
scale 缩放比例: 想把图片进行什么样的缩放.

'''
#M = np.float32([[1,0,100],[0,1,0]])
#M = cv2.getRotationMatrix2D((w/2,h/2),15,1.0)
# M = cv2.getRotationMatrix2D((100, 100), 45, 1.0)

'''
getAffineTransform(src[], dst[]) 通过三点可以确定变换后的位置, 
相当于解方程, 3个点对应三个方程, 能解出偏移的参数和旋转的角度.
- src原目标的三个点
- dst对应变换后的三个点
'''
# 平移操作
# 注意opencv中是先宽度, 后高度.
src = np.float32([[200, 100], [300, 100], [200, 300]])
dst = np.float32([[100, 150], [360, 200], [280, 120]])

M = cv2.getAffineTransform(src,dst)

new_dog = cv2.warpAffine(dog, M, dsize=(w,h))

cv2.imshow('dog',dog)
cv2.imshow('new_dog',new_dog)

cv2.waitKey(0)
cv2.destroyAllWindows()


