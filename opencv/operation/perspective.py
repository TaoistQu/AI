#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/31 15:48
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : perspective.py
# @Software: PyCharm
# description: 透视变换就是将一种坐标系变换成另一种坐标系.
# 简单来说可以把一张"斜"的图变"正".



import cv2
import numpy as np
import os

path = os.path.abspath('../images')
img = cv2.imread(os.path.join(path, './123.png'))

'''
warpPerspective(img, M, dsize,....)
对于透视变换来说, M是一个3 * 3 的矩阵.
 getPerspectiveTransform(src, dst) 获取透视变换的变换矩阵, 需要4个点, 即图片的4个角. 
'''
src = np.float32([[100, 1100], [2100, 1100], [0, 4000], [2500, 3900]])
dst = np.float32([[0, 0], [2300, 0], [0, 3000], [2300, 3000]])
M = cv2.getPerspectiveTransform(src, dst)

new_img = cv2.warpPerspective(img, M, (2300, 3000))

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 480, 640)
cv2.imshow('img', img)
cv2.imwrite(os.path.join(path,'math.png'), new_img)

cv2.namedWindow('new_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('new_img', 480, 640)
cv2.imshow('new_img', new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

