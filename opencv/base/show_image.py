#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/23 0:07
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : show_image.py
# @Software: PyCharm
# description:

import os,sys
sys.path.append(os.pardir)

from utils.utils import cv_show
import matplotlib.pyplot as plt
import cv2

cat = cv2.imread('D:\MyCode\AI\opencv\images\cat.jpeg')
# 发现matplotlib显示的图片和真实的图片颜色不一样. 因为opencv读进来的图片数据的通道不是默认的RGB
# 而是BGR, 所以一般opencv读进来的图片不用要别方式去展示比如matplotlib
# 用opencv自己的方式去展示就没有问题.

plt.imshow(cat)
plt.show()

cv_show('cat',cat)

