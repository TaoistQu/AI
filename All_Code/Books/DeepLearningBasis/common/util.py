#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/6 16:42
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : util.py
# @Software: PyCharm

import numpy as np

def im2col(input_data,filter_h,filter_w,stride=1,pad=0):
    """

       Parameters
       ----------
       input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
       filter_h : 滤波器的高
       filter_w : 滤波器的长
       stride : 步幅
       pad : 填充

       Returns
       -------
       col : 2维数组
       """
    N,C,H,W = input_data.shape
    out_h = (H+2*pad - filter_h) // stride +1
    out_w = (W + 2*pad - filter_w) //stride +1

    img = np.pad(input_data,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
    col = np.zeros(N,C,filter_h,filter_w,out_h,out_w)

    for y in range(filter_h):
        y_max = y+stride*out_h
        for x in range(filter_w):
            x_max = x+stride*out_w
            col[:,:,y,x,:,:] = img[:,:,y:y_max:stride,x:x_max:stride]

    return col


def smooth_curve(x):
    """用于使损失函数的图形变圆滑

      参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
      """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len,2)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[5:len(y)-5]








