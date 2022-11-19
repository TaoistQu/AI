#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/19 23:07
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : hfilter.py
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt



img=cv2.imread("D:\MyCode\AI\opencv\images\lp1.jpg",0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows,cols = img.shape

crow,ccol = int(rows/2),int(cols/2)

fshift[crow-30:crow+30,ccol-30:ccol+30] = 0

ishift = np.fft.fftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)

plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title("origin")
plt.axis('off')

plt.subplot(122)

plt.imshow(iimg,cmap='gray')
plt.title('result')
plt.axis('off')
plt.show()




