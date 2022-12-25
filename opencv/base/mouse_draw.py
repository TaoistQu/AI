#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/25 13:47
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : mouse_draw.py
# @Software: PyCharm
# description:
#  按下按键比如l, 进入绘制直线模式. 需要记录起始位置, 即按下鼠标左键的那一瞬间的坐标位置.
#  左键起来的鼠标坐标作为终点. 然后绘制.
#
# #%%
#
# # 按下l, 拖动鼠标, 可以绘制直线.
# # 按下r, 拖到鼠标, 可以绘制矩形
# # 按下c, 拖动鼠标, 可以绘制圆. 拖动的长度可以作为半径.

import cv2
import numpy as np

curshape = 0
startpos = (0, 0)

img = np.zeros((800, 800, 3), np.uint8)

def mouse_callback(event, x, y, flags, userdata):
    #引入全局变量
    global curshape, startpos
    if event == cv2.EVENT_LBUTTONDOWN:
        startpos = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        if curshape == 0:
            cv2.line(img,startpos,(x,y),(0,255,0),3)
        elif curshape == 1:
            cv2.rectangle(img,startpos,(x,y),(0,255,0),3)
        elif curshape == 2:
            #计算半径
            a = (x - startpos[0])
            b = (y - startpos[1])
            r = int((a**2+b**2)**0.5)
            cv2.circle(img,startpos,r,(0,235,200),3)
        else:
            print('暂不支持绘制其他图形')

cv2.namedWindow('drawshape',cv2.WINDOW_NORMAL)
cv2.resizeWindow('drawshape',640,480)
cv2.setMouseCallback('drawshape',mouse_callback)

while True:
    cv2.imshow('drawshape',img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('l'):
        curshape = 0
    elif key == ord('r'):
        curshape = 1
    elif key == ord('c'):
        curshape = 2

cv2.destroyAllWindows()



