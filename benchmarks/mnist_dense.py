# '''Trains a simple convnet on the MNIST dataset.
#
# Gets to 99.25% test accuracy after 12 epochs
# (there is still a lot of margin for parameter tuning).
# 16 seconds per epoch on a GRID K520 GPU.
# '''
#
from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np

# from pathlib import  Path
# import sys
# import argparse
#
# parser = argparse.ArgumentParser(description='Simple parameterized Deep Neural Network.')
#
# requiredNamed = parser.add_argument_group('required named arguments')
# requiredNamed.add_argument('-b', '--batch-size', help='Batch size',type=int, required=True)
# requiredNamed.add_argument('-n', '--neurons', help='Number of neurons per layer',type=int, required=True)
# requiredNamed.add_argument('-c', '--dense-count', help='Number of dense layers',type=int, required=True)
#
#
# args = parser.parse_args()
#
#
# batch_size = args.batch_size
# dense_size = args.neurons
# dense_count = args.dense_count
#
# print(f"batch-size: {batch_size}")
# print(f"neurons: {dense_size}")
# print(f"dense_count: {dense_count}")
#
# num_classes = 10
# epochs = 10
#
# # input image dimensions
# img_rows, img_cols = 28, 28
#
# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# model = Sequential()
# # model.add(Conv2D(32, kernel_size=(3, 3),
# #                  activation='relu',
# #                  input_shape=input_shape))
# # model.add(Conv2D(64, (3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))
#
# model.add(Flatten())
# for i in range(dense_count):
#     model.add(Dense(dense_size, activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test)
#           )
# # score = model.evaluate(x_test, y_test, verbose=0)
# # print('Test loss:', score[0])
# # print('Test accuracy:', score[1])
from tensorflow.python.keras import Input

# print(keras.__version__)

batch_size = 128
num_classes = 10
epochs = 5  # 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# #model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

model.add(Input(shape=input_shape))
model.add(Flatten())

model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(2048, activation="relu"))


model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))


model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)

# flops = []
# data_size = np.product(input_shape) * batch_size
# for layer in model.layers:
#     if layer.get_weights():
#         w, b = layer.get_weights()
#         flop = (np.product(w.shape)** 1.5) * batch_size + np.product(b.shape) * batch_size
#     else:
#         flop = data_size
#
#     flops.append(flop)
# print("FLOPS", np.sum(flops) / 1_000_000_000)
# print("Energy", (np.sum(flops) * 1.873243) / 1_000_000_000)


model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
)
score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
