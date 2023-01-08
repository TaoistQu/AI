#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2023/1/4 0:15
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : laplace.py
# @Software: PyCharm
# description:laplace是求图像的二阶导

import os
import cv2
import numpy as np

path = os.path.abspath('../images')

#img = cv2.imread(os.path.join(path,'lena_color.png'))
img = cv2.imread(os.path.join(path, 'chess.png'))

new_img = cv2.Laplacian(img, -1, ksize=5)

cv2.imshow('img', np.hstack((img, new_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()