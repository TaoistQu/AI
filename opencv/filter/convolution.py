#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/31 16:32
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : convolution.py
# @Software: PyCharm
# description:
#
# filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])
#
# - ddepth是卷积之后图片的位深, 即卷积之后图片的数据类型, 一般设为-1, 表示和原图类型一致.
# - kernel是卷积核大小, 用元组或者ndarray表示, 要求数据类型必须是float型.
# - anchor 锚点, 即卷积核的中心点, 是可选参数, 默认是(-1,-1)
# - delta 可选参数, 表示卷积之后额外加的一个值, 相当于线性方程中的偏差, 默认是0.
# - borderType 边界类型.一般不设.

import os
import cv2
import numpy as np

path = os.path.abspath('../images')

dog = cv2.imread(os.path.join(path,'./dog.jpeg'))

# kernel = np.ones((3, 3), np.float32) / 9

# 尝试其他卷积核, 突出轮廓
# kernel = np.array([[-1,-1, -1], [-1, 8, -1], [-1, -1, -1]])
#浮雕效果
#kernel = np.array([[-2, 1, 0], [-1, 1, 1], [0, 1, 2]])
#锐化
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

dst = cv2.filter2D(dog, -1, kernel)

print(dog.shape)
print(dst.shape)
cv2.imshow('dog', np.hstack((dog, dst)))
cv2.waitKey(0)
cv2.destroyAllWindows()



