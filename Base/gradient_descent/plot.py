#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/1 0:57
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : plot.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
f = lambda x :(x-3.5)**2 - 4.5*x + 10
x = np.linspace(-3,15,100)
y = f(x)
plt.plot(x,y,color='red')

plt.show()

g = lambda x : 2*(x-3.5) -4.5



