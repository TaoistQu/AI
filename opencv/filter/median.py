#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/1 12:53
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : median.py
# @Software: PyCharm
# description:中值滤波

import cv2
import os
import numpy as np

path = os.path.abspath('../images')

img = cv2.imread(os.path.join(path, './papper.png'))

new_img = cv2.medianBlur(img, 5)

cv2.imshow('img', np.hstack((img, new_img)))

cv2.waitKey(0)
cv2.destroyAllWindows()
