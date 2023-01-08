#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/5 23:12
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : threshold.py
# @Software: PyCharm
# description:
import numpy as np
import os
import cv2

path = os.path.abspath('../images')
img = cv2.imread(os.path.join(path, '8841.jpg'))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#返回两个结果，一个阈值，另一个处理后的图片
#ret, dst = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
#ret, dst = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
ret, dst = cv2.threshold(gray, 80, 255, cv2.THRESH_TRUNC)
cv2.imshow('dog', np.hstack((gray, dst)))
cv2.waitKey(0)
cv2.destroyAllWindows()

