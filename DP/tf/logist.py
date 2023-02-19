#自己实现算法的三步

#y = wx + b

import tensorflow as tf
from sklearn.datasets import make_blobs

import numpy as np

import matplotlib.pyplot as plt

data,target = make_blobs(centers=2)
plt.scatter(data[:,0],data[:,1],c=target)
plt.show()

#准备数据，一般加载数据
x = tf.constant(data,dtype=tf.float32)
y = tf.constant(target,dtype=tf.float32)

#初始化参数
W = tf.Variable(np.random.randn(2,1)*0.01 ,dtype=tf.float32)
B = tf.Variable(0. , dtype=tf.float32)
print(W.shape)
#预测函数
def sigmoid(x):
    linear = tf.matmul(x, W) + B
    #return 1 / (1 + tf.exp(-linear))
    return tf.nn.sigmoid(linear)

#损失函数
def cross_entropy_loss(y_true , y_pred):
    y_pred = tf.reshape(y_pred,shape=[100])
    return tf.reduce_mean(-(tf.multiply(y_true , tf.math.log(y_pred)) + tf.multiply((1-y_true),tf.math.log(1-y_pred))))


#训练过程

#定义优化器

optimizer = tf.optimizers.Adam()

def run_optimization():
    with tf.GradientTape() as g:
        y_pred = sigmoid(x)
        loss = cross_entropy_loss(y, y_pred)
        #计算梯度
        gradients = g.gradient(loss,[W,B])
        #更新
        optimizer.apply_gradients(zip(gradients,[W,B]))

def accuracy(y_true , y_pred):
    y_pred = tf.reshape(y_pred ,shape=[100])
    y_ = y_pred.numpy() >0.5
    y_true = y_true.numpy()
    return (y_ == y_true).mean()


for i in range(5000):
    run_optimization()

    if i % 100 == 0:
        y_pred = sigmoid(x)
        acc = accuracy(y , y_pred)
        loss = cross_entropy_loss(y , y_pred)
        print(f'训练次数:{i},准确率：{acc:.3f}，损失：{loss:.4f}')



