# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :Ptorch.py
# @Time      :2023/3/26 下午3:28
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:
import torch
import tensorflow as tf
import numpy as np


print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

#tensor

tensor = torch.tensor([1, 2, 3])

print(tensor)

ones = np.ones((3, 4))
print(ones)
torch_ones = torch.ones(3, 4)
print(torch_ones)

t1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

#tensor的运算加减乘除

#tensor的索引、切片、变形

t = torch.randint(0, 10, size=(4, 5))

print(t)

#切片 左闭右开
print(t[0, 0])
print(t[0])

print(t[1:3, 2:4])
print(t[:, 2:4])

n = np.random.random((32,224,224,3))
print(n)

t1 = torch.tensor(n)

print(t1)

t2 = t1[0,:,:,0]
print(t2.shape)

#聚合运算

print(t2.sum())
print(t2.sum())


tensor1 = torch.tensor([[1, 2, 3, 4, 5],
                        [2, 3, 4, 5, 6],
                        [3, 4, 11, 6, 7],
                        [4, 5, 6, 7, 8]])

tensor2 = torch.randint(0, 10, size=(4, 5))
print(tensor2)

print(tensor1.sum())

print(tensor1.sum(1))
print(tensor1.sum(0))

print(tensor1.argmax(0))
print(tensor1.argmax(1))

print(tensor2.argmax(0))

#拼接

t3 = torch.concat((tensor1, tensor2))
print(t3)

t4 = torch.concat((tensor1, tensor2), dim=1)
print(t4)

print(torch.hstack((tensor1, tensor2)))
print(torch.vstack((tensor1, tensor2)))
print(tensor1.split([1, 2, 1]))
#张量的微分

x = torch.ones(2, 2, requires_grad=True)
y = 2*x + 2

z = y.mean()

z.backward()


print(x.grad)






