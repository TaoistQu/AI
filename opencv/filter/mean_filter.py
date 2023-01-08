#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/1 12:31
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : mean_filter.py
# @Software: PyCharm
# description:
import cv2
import os
import numpy as np

path = os.path.abspath('../images')
dog = cv2.imread(os.path.join(path, './dog.jpeg'))

kernel = np.ones((5, 5), np.float32) / 25

con_dog = cv2.filter2D(dog, -1, kernel)
new_dog = cv2.blur(dog, (5, 5))

cv2.imshow('dog', np.hstack((dog, con_dog, new_dog)))

cv2.waitKey(0)
cv2.destroyAllWindows()
