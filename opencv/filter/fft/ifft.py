#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/19 21:23
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : ifft.py
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.abspath('../../images')

o = cv2.imread(os.path.join(path,'./lp.bmp'))
f = np.fft.fft2(o)

fshift = np.fft.fftshift(f)
ishift = np.fft.ifftshift(fshift)

io=np.fft.ifft2(ishift)
io = np.abs(io)

plt.subplot(121)
plt.imshow(o,cmap='gray')
plt.title('origin')
plt.axis('off')

plt.subplot(122)
plt.imshow(io.astype('uint8'),cmap='gray')
plt.title('result')
plt.axis('off')
plt.show()