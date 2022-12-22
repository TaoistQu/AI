#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/22 22:52
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : open_window.py
# @Software: PyCharm
# description:
import os,sys
sys.path.append(os.pardir)
from utils.utils import cv_show
import cv2

#创建窗口

cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.resizeWindow('window',600,400)
cv_show('window',0)
