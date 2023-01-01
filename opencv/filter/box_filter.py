#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/31 17:26
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : box_filter.py
# @Software: PyCharm
# description: 方盒滤波

import numpy as np
import cv2
import os

path = os.path.abspath('../images')
dog = cv2.imread(os.path.join(path,'./dog.jpeg'))

print(dog.shape)

# 不用手动创建卷积核, 只需要告诉方盒滤波, 卷积核的大小是多少.

dst = cv2.boxFilter(dog, -1, (3,3),normalize=True)

cv2.imshow('dst', np.hstack((dog, dst)))
cv2.waitKey(0)
cv2.destroyAllWindows()



