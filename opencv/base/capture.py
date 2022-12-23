#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 14:56
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : capture.py
# @Software: PyCharm
# description:

import cv2

cv2.namedWindow('video',cv2.WINDOW_NORMAL)
cv2.resizeWindow('video',640,480)

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()

    if not ret:
        break
    cv2.imshow('video',frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        print('关闭视频')
        break

cap.release()
cv2.destroyAllWindows()