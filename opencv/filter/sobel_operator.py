#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/1 19:52
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : sobel_operator.py
# @Software: PyCharm
# description: 边缘是像素值发生跃迁的位置，是图像的显著特征之一
# sobel算子对图像求一阶导数。一阶导数越大，说明像素在该方向的变化越大，边缘信号越强。

import os
import numpy as np
import cv2

path = os.path.abspath('../images')

chess = cv2.imread(os.path.join(path,'chess.png'))

dx = cv2.Sobel(chess, cv2.CV_64F, dx=1, dy=0, ksize=3)
dy = cv2.Sobel(chess, cv2.CV_64F, dx=0, dy=1, ksize=3)

new_img = cv2.add(dx, dy)

cv2.imshow('img',np.hstack((dx,dy)))
cv2.imshow('new_img', np.hstack((chess,new_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()

