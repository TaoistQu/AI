#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/9 23:45
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : l1.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

# α 的值是：1
# 1 = x + y
# y = 1 -x
f = lambda x : 1- x
f4 = lambda x : x - 1
f5 = lambda x : -x -1
f6 = lambda x : x + 1
x = np.linspace(0,1,100)
x4 = np.linspace(-1,0,100)
plt.axis('equal')
plt.plot(x, f(x), color = 'green')
plt.plot(x4, f5(x4), color = 'green')
plt.plot(x,f4(x),color='green')
plt.plot(x4,f6(x4),color='green')

# α 的值是：3
# 1 = 3 * x + 3 * y
# y = 1/3 -x
f2 = lambda x : 1/3 - x
f3 = lambda x : x -1/3
x2 = np.linspace(0,1/3,100)
#x3 = np.linspace(-1/3,0,100)
plt.plot(x2, f2(x2),color = 'red')
plt.plot(x2, f3(x2),color = 'red')

# 一些列设置
plt.xlim(-2,2)
plt.ylim(-2,2)
ax = plt.gca()
ax.spines['right'].set_color('None')  # 将图片的右框隐藏
ax.spines['top'].set_color('None')  # 将图片的上边框隐藏
ax.spines['bottom'].set_position(('data', 0)) # x轴出现在y轴的-1 位置
ax.spines['left'].set_position(('data', 0))
plt.show()
plt.savefig('D:\MyCode\AI\Base\image\lpha.png',dpi = 200)
