#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/22 1:19
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : low_filter.py
# @Software: PyCharm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

path = os.path.abspath('../../images')
img = cv2.imread(os.path.join(path,"./lp1.jpg"), 0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)
rows,cols= img.shape
crow,ccol = int(rows/2),int(cols/2)
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30,ccol-30:ccol+30] = 1
fShift = dftShift * mask
ishift = np.fft.ifftshift(fShift)
iImg = cv2.idft(ishift)
iImg = cv2.magnitude(iImg[:,:,0],iImg[:,:,1])
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title('original')
plt.axis('off')
plt.subplot(122)
plt.imshow(iImg,cmap='gray')
plt.title('inverse')
plt.axis('off')
plt.show()