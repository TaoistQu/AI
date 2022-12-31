#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/31 14:07
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : rotate.py
# @Software: PyCharm
# description:# - ROTATE_90_CLOCKWISE 90度顺时针
# # - ROTATE_180 180度
# # - ROTATE_90_COUNTERCLOCKWISE 90度逆时针

import cv2
import os
import numpy
import numpy as np

path = os.path.abspath('../images')

dog = cv2.imread(os.path.join(path, '../images/dog.jpeg'))
print(dog.shape)
dst = np.zeros((499,360))
new_dog = cv2.rotate(dog, cv2.ROTATE_90_CLOCKWISE, dst)
cv2.imshow('dog',dog)
cv2.imshow('new_dog',new_dog)
cv2.imshow('dst',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
