import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
#正标准化
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

#one hot

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
#同级数据的缝补
#plt.hist(X_test_scaled, bins=30, range=[-2.5, 2.5])
#

#丁酉网络

model = keras.Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

summary = model.summary()
print(summary)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train_scaled, y_train, batch_size=64, epochs=20, validation_data=(X_test_scaled, y_test))
print(history.history)

pd.DataFrame(history.history).plot()
plt.show()
