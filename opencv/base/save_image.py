#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 10:53
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : save_image.py
# @Software: PyCharm
# description:
import cv2

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',640,480)

img = cv2.imread('D:\MyCode\AI\opencv\images\cat.jpeg')

while True:
    cv2.imshow('img',img)
    key = cv2.waitKey(0)

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('123.png',img)
    else:
        print(key)

cv2.destroyAllWindows()