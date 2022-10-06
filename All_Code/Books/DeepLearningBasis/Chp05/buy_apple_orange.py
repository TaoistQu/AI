#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/6 15:36
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : buy_apple_orange.py
# @Software: PyCharm

from layer_naive import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_organe_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward

apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
all_price = add_apple_organe_layer.forward(apple_price,orange_price)
price = mul_tax_layer.forward(all_price,tax)

#backward

dprice =1

dall_price,dtax = mul_tax_layer.backward(dprice)
dapple_price,dorange_price = add_apple_organe_layer.backward(dall_price)
dorange,dorange_num = mul_orange_layer.backward(dorange_price)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)


print("price:", int(price))
print("DAprice:",dall_price)
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dOrange:", dorange)
print("dOrange_num:", int(dorange_num))
print("dTax:", dtax)





