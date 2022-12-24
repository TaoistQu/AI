#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 22:06
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : color_transfer.py
# @Software: PyCharm
# description:
import cv2
import os

def callback(value):
    print(value)

cv2.namedWindow('color',cv2.WINDOW_NORMAL)
cv2.resizeWindow('color',640,480)

path = os.path.abspath('../images')
img = cv2.imread(os.path.join(path,'cat.jpeg'))

color_spaces = [
    cv2.COLOR_BGR2RGBA,cv2.COLOR_BGR2BGRA,
    cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV,
    cv2.COLOR_BGR2YUV
]

#设置trackBar
cv2.createTrackbar('trackbar','color',0,4,callback)

while True:
    index = cv2.getTrackbarPos('trackbar','color')
    #进行颜色空间转换
    cvt_img = cv2.cvtColor(img,color_spaces[index])

    cv2.imshow('color', cvt_img)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
