#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/5 23:29
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : adaptive_threshold.py
# @Software: PyCharm
# description:自适应阈值二值化

import cv2
import numpy as np

img = cv2.imread('./images/math.png')
dog = cv2.imread('./images/dog.jpeg')
# 二值化操作是对灰度图像操作, 把dog变成灰度图像
gray_dog = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
gray_math = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 自适应阈值二值化只有一个返回值
dst_dog = cv2.adaptiveThreshold(gray_dog, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 19, 0)

dst_math = cv2.adaptiveThreshold(gray_math, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 3, 0)

cv2.imshow('dog', np.hstack((gray_dog, dst_dog)))
cv2.namedWindow('math', cv2.WINDOW_NORMAL)
cv2.resizeWindow('math', 640, 480)
cv2.imshow('math', gray_math)

cv2.namedWindow('dst_math', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dst_math', 640, 480)
cv2.imshow('dst_math', dst_math)
cv2.imwrite('./images/dst_math.png', dst_math)
cv2.waitKey(0)
cv2.destroyAllWindows()


