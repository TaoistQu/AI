#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 22:39
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : split_chanl.py
# @Software: PyCharm
# description:

import cv2
import numpy as np

img = np.zeros((200,200,3),np.uint8)

b,g,r = cv2.split(img)

b[10:100,10:50] = 255
g[10:100,10:100] = 255

#合并通道
img2 = cv2.merge((b,g,r))

cv2.imshow('img', np.hstack((b,g)))
cv2.imshow('img2', np.hstack((img,img2)))

cv2.waitKey(0)
cv2.destroyAllWindows()