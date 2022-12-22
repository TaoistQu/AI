#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/22 22:52
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : img_read.py
# @Software: PyCharm
# description:

import cv2

#创建窗口

cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.resizeWindow('window',600,400)
cv2.imshow('window',0)
key = cv2.waitKey(0)
if key == ord('q'):
    print('准备销毁窗口')
    cv2.destroyAllWindows()

