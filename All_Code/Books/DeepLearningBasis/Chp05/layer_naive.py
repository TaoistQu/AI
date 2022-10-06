#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/6 15:20
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : layer_naive.py
# @Software: PyCharm


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None


    def forward(self,x,y):
        self.x  = x
        self.y = y
        out = x*y
        return out

    def backward(self,dout):
        dx = dout*self.y
        dy = dout*self.x
        return dx,dy

class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        out = x+y
        return out

    def backward(self,dout):
        dx = dout*1
        dy = dout*1

        return dx,dy