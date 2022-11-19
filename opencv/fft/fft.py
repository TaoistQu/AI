#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/19 20:49
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : fft.py
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt

o = cv2.imread("D:\MyCode\AI\All_Code\opencv\images\lena.bmp",0)


f = np.fft.fft2(o)
fshift = np.fft.fftshift(f)
resut = 20*np.log(np.abs(fshift))
plt.subplot(121)
plt.imshow(o,cmap='gray')
plt.title('orginal')
plt.axis('off')
plt.subplot(122)
plt.imshow(resut,cmap='gray')
plt.title('result')
plt.axis('off')
plt.show()