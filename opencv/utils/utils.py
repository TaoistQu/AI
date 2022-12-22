#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 0:12
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : utils.py
# @Software: PyCharm
# description:
import cv2
def cv_show(name,img):
    cv2.imshow(name,img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        print('准备销毁窗口')
        cv2.destroyAllWindows()