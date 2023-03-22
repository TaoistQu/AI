# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :transfer.py
# @Time      :2023/3/22 下午8:59
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:使用迁移学习训练网络

from keras import Sequential
from keras.applications.resnet import ResNet50
from keras.layers import Dense
import keras
from keras.preprocessing.image import ImageDataGenerator

resnet50 = ResNet50(include_top=False, classes=10, pooling='avg')
model = Sequential()
model.add(resnet50)
model.add(Dense(10, activation='softmax'))

model.summary()

#固定某层不训练

model.layers[0].trainable = False

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

train_dir  = '../data/training/training'
valid_dir = '../data/validation/validation'
label_file ='../data/monkey_labels.txt'

def main():
    # 生成数据，使用迁移学习时对数据预处理，要对数据预处理采用相同的方式
    train_datagen = ImageDataGenerator(
        preprocessing_function= keras.applications.resnet.preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )


    # 从目录中读取图片
    height = 224
    width = 224
    channels = 3
    batch_size = 64
    num_classes = 10

    # 会自动把目录名作为label名.
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(
        height, width),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=7,
                                                        class_mode='categorical')

    valid_datagen = ImageDataGenerator(
        preprocessing_function=keras.applications.resnet.preprocess_input
    )

    valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(
        height, width),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical')


    train_num = train_generator.samples
    valid_num = valid_generator.samples

    history = model.fit(train_generator,
                          steps_per_epoch=train_num // batch_size,
                          epochs=30,
                          validation_data=valid_generator,
                          validation_steps=valid_num // batch_size)


    model.save('./ResNet50.h5')


if __name__ == '__main__':
    main()
