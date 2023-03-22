# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :predict.py
# @Time      :2023/3/22 下午9:57
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:预测 预测的时候也需要按照训练的维度： batch_size, W,H,C


import keras
import cv2
import matplotlib.pyplot as plt
from keras.applications.resnet import preprocess_input

model = keras.models.load_model('./ResNet50.h5')
model.summary()




def predict(image_path):
    img = plt.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_data = img.reshape(1, 224, 224, 3)

    img_data = preprocess_input(img_data)
    result = model.predict(img_data)
    print(result.argmax(axis=1))


predict('../data/validation/validation/n7/n700.jpg')


