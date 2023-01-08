#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/31 14:01
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : flip.py
# @Software: PyCharm
# description:# flipCode=0, 表示上下翻转
# # flipCode>0, 表示左右翻转
# # flipCode <0, 表示上下左右翻转


import cv2
import numpy as np
import os

path = os.path.abspath('../images')
dog = cv2.imread(os.path.join(path, './dog.jpeg'))

new_dog = cv2.flip(dog, 1)

cv2.imshow('img', np.hstack((dog, new_dog)))
cv2.waitKey(0)
cv2.destroyAllWindows()
