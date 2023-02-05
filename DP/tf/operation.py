import tensorflow as tf
import numpy as np

a = tf.constant(np.random.randint(0, 10, size=(3, 4)))
b = tf.constant(np.random.randint(0, 10, size=(4, 5)))

c = tf.matmul(a, b)

print(c)
#实现聚合
d = tf.reduce_sum(c)
print(d)

