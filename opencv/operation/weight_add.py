#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/25 18:19
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : weight_add.py
# @Software: PyCharm
# description:
import cv2
import os
import numpy as np

path = os.path.abspath('../images')
cat = cv2.imread(os.path.join(path, './cat.jpeg'))
dog = cv2.imread(os.path.join(path, './dog.jpeg'))

cat = cat[:360, :499]

#按比例融合图片
img_new = cv2.addWeighted(dog, 0.5, cat, 0.5, 0)
cv2.imshow('new_img', np.hstack((dog, cat, img_new)))

cat_not = cv2.bitwise_not(cat)
cv2.imshow('not', np.hstack((cat, cat_not)))

img_and = cv2.bitwise_and(dog, cat)
cv2.imshow('and', np.hstack((cat, dog, img_and)))
cv2.waitKey(0)
cv2.destroyAllWindows()


