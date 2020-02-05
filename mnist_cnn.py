'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from pathlib import  Path
import sys
from datetime import datetime
import monkey_patch

data_root = Path(sys.argv[-1])
timestamp_log_path = data_root / "timestamps.csv"


batch_size = int(sys.argv[-2])
dense_size = int(sys.argv[-3])
dense_count = int(sys.argv[-4])
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# timestamp logging shit
timestamp_log = open(timestamp_log_path ,"w",buffering=1)
timestamp_log.write(f"timestamp,event,data\n")

def train_cb(logs):
    #print("log begin")
    timestamp_log.write(f"{str(datetime.now())},train_begin,\n")

def epoch_cb(epoch,logs):
    #print("log begin_epoch")
    timestamp_log.write(f"{str(datetime.now())},epoch_begin,{epoch}\n")

logger = keras.callbacks.LambdaCallback(
            on_epoch_begin=epoch_cb ,
            on_epoch_end=lambda epoch, logs: timestamp_log.write(f"{str(datetime.now())},epoch_end,{epoch}\n"),
            on_train_begin=train_cb,
            on_train_end=lambda logs: timestamp_log.write(f"{str(datetime.now())},train_end,\n"),
            on_batch_begin=lambda epoch, logs: timestamp_log.write(f"{str(datetime.now())},batch_begin,{epoch}\n")   )



# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
for i in range(dense_count):
    model.add(Dense(dense_size, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)
          #,callbacks=[logger]
          )
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])