#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/4 0:22
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : canny.py
# @Software: PyCharm
# description:

import os
import numpy as np
import cv2

path = os.path.abspath('../images')

img = cv2.imread(os.path.join(path, 'lena_color.png'))
new_img = cv2.Canny(img, 100, 200)
img1 = cv2.Canny(img, 64, 128)

cv2.imshow('img', np.hstack((img1, new_img)))
#cv2.imshow('img',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
