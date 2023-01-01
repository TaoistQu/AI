#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/1 19:33
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : bilateral_filter.py
# @Software: PyCharm
# description:双边滤波

import os
import cv2
import numpy as np

path = os.path.abspath('../images')
img = cv2.imread(os.path.join(path, './lena_color.png'))
'''
bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])
sigmaColor是计算像素信息使用的sigma
sigmaSpace是计算空间信息使用的sigma

'''
new_img = cv2.bilateralFilter(img, 7, 20, 50)
#cv2.namedWindow('img',cv2.WINDOW_NORMAL)
print(img.shape)
#cv2.resizeWindow('img', 1706, 1280)
cv2.imshow('img', np.hstack((img, new_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()

