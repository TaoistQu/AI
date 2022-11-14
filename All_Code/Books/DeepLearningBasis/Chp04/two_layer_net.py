#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/10/5 10:22
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : two_layer_net.py
# @Software: PyCharm
import numpy as np
import os
import sys
from common.functions import *
from common.gradient import numerical_gradient

class TowLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.parmas = {}
        self.parmas['W1'] = weight_init_std*np.random.randn(input_size,hidden_size)
        self.parmas['b1'] = np.random.randn(hidden_size)
        self.parmas['W2'] = weight_init_std*np.random.randn(hidden_size,output_size)
        self.parmas['b2'] = np.zeros(output_size)


    def predict(self,x):
        W1,W2 = self.parmas['W1'],self.parmas['W2']
        b1,b2 = self.parmas['b1'],self.parmas['b2']

        a1 = np.dot(x,W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2)+b2
        y = softmax(a2)

        return y

    def loss(self,x,t):
        y = self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W:self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.parmas['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.parmas['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.parmas['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.parmas['b2'])

        return grads


    def gradient(self,x,t):
        W1,W2 = self.parmas['W1'],self.parmas['W2']
        b1,b2 = self.parmas['b1'],self.parmas['b2']
        grads = {}

        batch_num = x.shape[0]

        #forward
        a1 = np.dot(x,W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        #backward
        dy = (y-t) /batch_num
        grads['W2'] = np.dot(z1.T,dy)
        grads['b2'] = np.sum(dy,axis=0)

        da1 = np.dot(dy,W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T,dz1)
        grads['b1'] = np.sum(dz1,axis=0)

        return grads