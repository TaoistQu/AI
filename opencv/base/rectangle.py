#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 22:51
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : rectangle.py
# @Software: PyCharm
# description:

import cv2
import numpy as np

img = np.zeros((480,640,3),np.uint8)

cv2.line(img,(10,20),(300,400),[0,0,255],5,4)
cv2.line(img,(80,100),(380,480),(0,255,0),5,16)
#画矩形
cv2.rectangle(img,(10,10),(200,200),(0,255,24),5,16)
cv2.circle(img,(200,200),100,(10,160,100),5,16)

cv2.imshow('draw',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
