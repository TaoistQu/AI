#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/6 18:28
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : two_layer_net.py
# @Software: PyCharm

from Books.DeepLearningBasis.common.layers import *
from Books.DeepLearningBasis.common.gradient import numerical_gradient
from collections import OrderedDict
import numpy as np

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):
       #初始化权重参数
        self.params = {}
        self.params['W1'] = weight_init_std* np.random.randn(input_size,hidden_size)
        self.params['b1'] = weight_init_std*np.random.randn(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
        #生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x  = layer.forward(x)

        return x

    #x:输入数据，t：监督数据
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)

        if t.ndim != 1 : t = np.argmax(t,axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    #x:输入数据，t：监督数据
    def numerical_gradient(self,x,t):
        loss_W = lambda W : self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads

    def gradient(self,x,t):
        #forward
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())

        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads={}
        grads['W1'],grads['b1'] = self.layers['Affine1'].dW,self.layers['Affine1'].db
        grads['W2'],grads['b2'] = self.layers['Affine2'].dW,self.layers['Affine2'].db

        return grads



