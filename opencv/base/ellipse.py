#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 23:10
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : ellipse.py
# @Software: PyCharm
# description:
import cv2
import numpy as np

img = np.zeros((480,640,3),np.uint8)
cv2.ellipse(img,(120,240),(100,50),0,0,360,(0,0,255),5,16)
pts = np.array([(100,100),(150,60),(180,90),(380,280),(200,360)],np.int32)
#pts是个三维的点集
cv2.polylines(img,[pts],True,(0,255,0),5,16)
cv2.fillPoly(img,[pts],(0,255,255))
cv2.imshow('draw',img)
cv2.waitKey(0)
cv2.destroyAllWindows()