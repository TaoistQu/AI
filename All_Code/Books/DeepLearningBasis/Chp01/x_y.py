#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/6 22:21
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : x_y.py
# @Software: PyCharm
#画三维曲面

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
def f(x, y):
   return 1/20*(x ** 2) + y ** 2
#构建x、y数据
x = np.linspace(-6, 6,30)

y = np.linspace(-1, 1, 30)
#将数据网格化处理
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
#50表示在z轴方向等高线的高度层级，binary颜色从白色变成黑色
ax.contour3D(X, Y, Z, 50, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')
plt.show()

