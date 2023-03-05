# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# @FileName  :data_file.py
# @Time      :2023/3/5 下午9:47
# @Author    :TaoistQu
# Email      :qulei_20180331@163.com
# description:

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from california import get_data

x_train_scaled, x_valid_scaled, x_test_scaled, y_train, y_valid, y_test, header_clos = get_data()
#生成文件

output_dir = 'generate_csv'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def save_to_csv(output_dir, data, name_prefix , header=None, n_parts=10):
    path_format = os.path.join(output_dir, '{}_{:02d}.csv')
    filenames = []
    for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)

        with open(part_csv,'wt', encoding='utf-8') as f:
            if header is not  None:
                f.write(header +'\n')
            for row_index in row_indices:
                f.write(','.join([repr(col) for col in data[row_index]]))
                f.write('\n')
    return filenames

#把样本数据和对应的标记合并在一起
train_data = np.c_[x_train_scaled, y_train]
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]

header_clos = header_clos + ['MedianHouseValue']
header_str = ','.join(header_clos)

train_filenames = save_to_csv(output_dir, train_data, 'train', header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data, 'valid', header_str, n_parts=20)
test_filenames = save_to_csv(output_dir, test_data, 'test', header_str, n_parts=20)

filename_datasets = tf.data.Dataset.list_files(train_filenames)

n_readers = 5

dataset = filename_datasets.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1),
    cycle_length= n_readers
)

for line in dataset.take(5):
    print(line.numpy())


#解析csv
def parse_csv_line(line, n_fields=9):
    record_defaults = [tf.constant(np.nan)]*n_fields
    parse_fields = tf.io.decode_csv(line, record_defaults)
    x = tf.stack(parse_fields[0:-1])
    y = tf.stack(parse_fields[-1:])
    return x, y

def parse_csv_line_origin(line, n_fileds=9):
    line = line.decode()
    data = [np.array(float(x)) for x in line.split(',')]
    x = np.stack(data[0:-1])
    y = np.stack(data[-1:])
    return x, y


def csv_reader_dataset(filenames, n_readers=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset
batch_size = 64
train_set = csv_reader_dataset(train_filenames,batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames,batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames,batch_size=batch_size)

model = Sequential([Dense(32, activation='relu', input_shape=[8]),
                    Dense(1)])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

history = model.fit(train_set,
                    validation_data=valid_set,
                    steps_per_epoch=len(x_train_scaled) // batch_size,
                    validation_steps= len(x_valid_scaled) // batch_size,
                    epochs=100)




