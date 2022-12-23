#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 16:12
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : mouse_event.py
# @Software: PyCharm
# description:
import cv2
import numpy as np

def mouse_callback(event,x,y,flags,userdata):
    print(event,x,y,flags,userdata)
    if event == 2:
        cv2.destroyAllWindows()

# 创建窗口
cv2.namedWindow('mouse',cv2.WINDOW_NORMAL)
cv2.resizeWindow('mouse',640,360)

cv2.setMouseCallback('mouse',mouse_callback,'123')

#生成一张全黑的图片
img = np.zeros((360,640,3),np.uint8)
while True:
    cv2.imshow('mouse',img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
