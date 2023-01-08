#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/25 19:07
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : logo.py
# @Software: PyCharm
# description:
import cv2
import os
import numpy as np

path = os.path.abspath('../images')

dog = cv2.imread(os.path.join(path, './dog.jpeg'))
logo = np.zeros((200, 200, 3), np.uint8)
#绘制logo
logo[20:120, 20:120] = [0, 0, 255]
logo[80:180, 80:180] = [0, 255, 0]
#设置掩码
mask = np.zeros((200, 200), np.uint8)
mask[20:120, 20:120] = 255
mask[80:180, 80:180] = 255
#cv2.imshow('mask', mask)

m = cv2.bitwise_not(mask)
#cv2.imshow('m', m)
#在原图中选择添加logo的位置
roi = dog[:200, :200]
# roi与m进行与操作, 先roi和roi做与运算, 然后结果再和mask做与运算,
# 如果与的结果是True, 那么返回原图的像素, 否则返回0
tmp = cv2.bitwise_and(roi, roi, mask=m)
dst = cv2.add(tmp, logo)
cv2.imshow('tmp', tmp)
cv2.imshow('dog1', dog)
dog[:200,:200] = dst


cv2.imshow('dog', dog)
#cv2.imshow('logo', logo)

cv2.waitKey(0)
cv2.destroyAllWindows()