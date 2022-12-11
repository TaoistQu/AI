#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/11 19:44
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : sigmoid_fig.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))

x= np.linspace(-5,5,100)
y = sigmoid(x)
plt.plot(x,y,color='green')
plt.title('sigmoid_fig')
plt.xlabel('x')
plt.ylabel('sigmoid')
plt.show()
