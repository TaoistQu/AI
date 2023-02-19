import keras
from keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
#from keras.datasets import mnist
from keras.datasets import fashion_mnist
import numpy as np

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

print(x_train.shape)

plt.imshow(x_train[1],cmap='gray')
plt.show()

#对数据进行预处理
#归一化数据
X_train = x_train.reshape(-1, 784) / 255.0
X_test = x_test.reshape(-1, 784) / 255.0



#对y进行one-hot 编码
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#定义神经网络

model = keras.Sequential()
#神经元的个数
#第一层
model.add(Dense(64, activation='relu', input_shape=(784,)))

#第二层

model.add(Dense(64, activation='relu'))
#输出层
model.add(Dense(10, activation='softmax'))
#配置网络
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#训练网络，一个epochs为完整遍历一次数据, 一步取batch_size个样本
model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))

#模型评估
eva = model.evaluate(X_test, y_test)
pre = model.predict(X_test[:32])[0].argmax()
print(eva)
print(pre)
#计算模型参数
sum = model.summary()
print(sum)

model.save('./model.h5')


