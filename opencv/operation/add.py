#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/25 17:46
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : add.py
# @Software: PyCharm
# description:
import os
import cv2
import numpy as np

path = os.path.abspath('../images')

cat = cv2.imread(os.path.join(path,'./cat.jpeg'))
dog = cv2.imread(os.path.join(path,'./dog.jpeg'))
#cat = cat[:360,:499]
cat = cv2.resize(cat,(499,360))

new_img = cv2.add(cat,dog)
#new_img = cv2.multiply(cat,dog)
#new_img = cv2.subtract(cat,dog)
#new_img = cv2.divide(cat,dog)

#cv2.imshow('dog',dog)
#cv2.imshow('cat',cat)

cv2.imshow('new_img',np.hstack((new_img,cat,dog)))

cv2.waitKey(0)
cv2.destroyAllWindows()