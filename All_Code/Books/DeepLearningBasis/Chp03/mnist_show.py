#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/3 19:53
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : mnist_show.py
# @Software: PyCharm
import os
import sys
from dataset import mnist
from PIL import Image
import numpy as np
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train,t_train),(x_test,t_test) = mnist.load_mnist(flatten=True,normalize=False)
img = x_train[0]
label = t_train[0]

print(label)

print(img.shape)

img = img.reshape(28,28)
print(img.shape)

img_show(img)






