#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 15:10
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : video.py
# @Software: PyCharm
# description:
import cv2
import os

cv2.namedWindow('video',cv2.WINDOW_NORMAL)
cv2.resizeWindow('video',640,480)

path = os.path.abspath('../video')

cap = cv2.VideoCapture(os.path.join(path,'./my.mp4'))

while True:
    ret,frame = cap.read()

    if not ret:
        break

    cv2.imshow('video',frame)

    key = cv2.waitKey(1000 // 30)
    if key == ord('q'):
        print('关闭')

cap.release()
cv2.destroyAllWindows()

