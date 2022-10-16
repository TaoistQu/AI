#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/6 22:49
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : equal.py
# @Software: PyCharm
#画等高线
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, num=100)
y = np.linspace(-1, 1, num=100)
X, Y = np.meshgrid(x, y)
print(x.shape)
#f = -X**2 - Y**2
f = 1/20*(X ** 2) + Y ** 2

fig = plt.figure()
plt.xlim(-6, 6)
plt.ylim(-1.5, 1.5)

# draw
ax = plt.contour(X, Y, f, levels=10, cmap=plt.cm.cool)
# add label
plt.clabel(ax, inline=True)
# plt.savefig('img1.png')
plt.show()

