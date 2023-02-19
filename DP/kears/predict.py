import keras
from keras.datasets import mnist
import numpy as np
import tensorflow as tf


(x_train,y_train),(x_test,y_test) = mnist.load_data()

#对数据进行预处理
#归一化数据
X_train = x_train.reshape(-1, 784) / 255.0
X_test = x_test.reshape(-1, 784) / 255.0



#对y进行one-hot 编码
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#模型加载
model = keras.models.load_model('model.h5')

pre = model.predict(X_test[:32])[0].argmax()
print(pre)