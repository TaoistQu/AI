#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/1 20:38
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : scharr_operator.py
# @Software: PyCharm
# description:

import cv2
import os
import numpy as np

path = os.path.abspath('../images')

img = cv2.imread(os.path.join(path,'lena_color.png'))

dx = cv2.Scharr(img, -1, dx=1, dy=0)
dy = cv2.Scharr(img, -1, dx=0, dy=1)
new_img = cv2.add(dx, dx)

cv2.imshow('img',np.hstack((img,new_img)))

cv2.waitKey(0)
cv2.destroyAllWindows()

