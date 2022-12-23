#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 15:46
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : capture_video.py
# @Software: PyCharm
# description:

import cv2
import os

cap = cv2.VideoCapture(0)

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
path = os.path.abspath('../video')
#vw = cv2.VideoWriter(os.path.join(path,'./output.mp4'),fourcc,30,(640,480))
vw = cv2.VideoWriter(os.path.join(path,'./output.avi'),fourcc,30,(640,480))

while cap.isOpened():
    ret ,frame = cap.read()
    if not ret:
        break

    vw.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1000 // 30) == ord('q'):
        break

cap.release()
vw.release()
cv2.destroyAllWindows()