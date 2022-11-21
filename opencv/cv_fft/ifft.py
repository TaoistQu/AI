#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/22 1:04
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : ifft.py
# @Software: PyCharm
import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("D:\MyCode\AI\opencv\images\lp1.jpg",0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)
ishift = np.fft.ifftshift(dftShift)

iImg = cv2.idft(ishift)
iImg = cv2.magnitude(iImg[:,:,0],iImg[:,:,1])
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(iImg,cmap='gray')
plt.title('result')
plt.axis('off')
plt.show()


