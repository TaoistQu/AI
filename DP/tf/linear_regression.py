import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import tensorflow as tf



x = np.linspace(0, 10, 20) + np.random.randn(20)
#x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20) + np.random.randn(20)
#y = np.linspace(0, 10, 20)



linear = LinearRegression()
linear.fit(x.reshape(-1, 1), y)
print(linear.coef_)

x_test = np.linspace(0,10,30)
y_test = linear.coef_[0] * x_test +linear.intercept_
plt.scatter(x, y, c='g')
plt.plot(x_test, y_test, c='r')
plt.show()

W = tf.Variable(np.random.randn()*0.01)
B = tf.Variable(0.)

def linear_regression(x):
    return W *x + B

def mean_square_loss (y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

optimizer = tf.optimizers.SGD()

def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(x)
        loss = mean_square_loss(pred,y)

    gradients = g.gradient(loss, [W,B])

    optimizer.apply_gradients(zip(gradients,[W,B]))

epochs = 5000
for epoch in range(epochs):
    run_optimization()

    if epoch % 100 == 0:
        pred = linear_regression(x)
        loss = mean_square_loss(pred,y)
        print(f'step:{epoch+1},loss:{loss},W:{W.numpy()},B:{B.numpy()}')




print(W.numpy())
print(linear.coef_)
print(B.numpy())
print(linear.intercept_)

