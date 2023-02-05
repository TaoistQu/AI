import numpy as np
import tensorflow as tf


#tensorflow中的常量
from tensorboard.notebook import display

a = tf.constant(1, dtype=tf.float32)
b = tf.constant([[1, 2, 3], [4, 5, 6]])

print(a)
print(b)

v = tf.Variable([[1,2,3],[4,5,6]],name='V',validate_shape=True)

#gpu_device_name = tf.test.gpu_device_name()
#print(gpu_device_name)

#print( tf.test.is_gpu_available())
print(v)


