#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 16:46
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : track_bar.py
# @Software: PyCharm
# description:
import cv2
import numpy as np

cv2.namedWindow('trackbar',cv2.WINDOW_NORMAL)
cv2.resizeWindow('trackbar',640,480)

#定义回调函数
def callback(value):
    print(value)

cv2.createTrackbar('R','trackbar',0,255,callback)
cv2.createTrackbar('G','trackbar',0,255,callback)
cv2.createTrackbar('B','trackbar',0,255,callback)

#创建背景图片
img = np.zeros((480,640,3),np.uint8)

while True:
    r = cv2.getTrackbarPos('R','trackbar')
    g = cv2.getTrackbarPos('G', 'trackbar')
    b = cv2.getTrackbarPos('B', 'trackbar')

    img[:] = [b,g,r]

    cv2.imshow('trackbar',img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
