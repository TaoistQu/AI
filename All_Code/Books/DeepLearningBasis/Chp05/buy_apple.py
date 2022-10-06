#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/6 15:19
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : buy_apple.py
# @Software: PyCharm

from layer_naive import *

apple = 100
apple_num = 2

tax  = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward

apple_price = mul_apple_layer.forward(apple,apple_num)
price = mul_tax_layer.forward(apple_price,tax)

#backward

dprice = 1
dapple_price,dtax = mul_tax_layer.backward(dprice)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)

print("price:",int(price))
print("dApple:",dapple)
print("dApple_num ：",int(dapple_num))
print("dTax: ",dtax)