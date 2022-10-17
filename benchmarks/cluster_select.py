from tensorflow import keras
import numpy as np


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60_000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# preprocessing the images
# convert each image to 1 dimensional array
X = x_train.reshape(len(x_train), -1)
Y = y_train

# normalize the data to 0 - 1
X = X.astype(float) / 255.0


from sklearn.cluster import MiniBatchKMeans

n_digits = len(np.unique(y_test))
print(n_digits)
n_clusters = 10
percentage = .9

selected_x = []
selected_y = []

for label_class in range(n_digits):
    # Initialize KMeans model
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)

    # Fit the model to the training data
    x_label = X[Y == label_class]
    kmeans.fit(x_label)
    y_hat = kmeans.predict(x_label)
    for cluster in range(n_clusters):
        errors = np.mean(
            (x_label[y_hat == cluster] - kmeans.cluster_centers_[cluster]) ** 2, axis=1
        )
        assert len(x_label[y_hat == cluster]) == len(errors), (len(x_label[y_hat == cluster]), len(errors))
        n_smallest = int((1-percentage) * len(errors))
        smallest_idx = np.argpartition(errors, n_smallest)[n_smallest:]
        selected_x.extend(x_label[y_hat == cluster][smallest_idx])
        selected_y.extend(Y[Y == label_class][y_hat == cluster][smallest_idx])

selected_x = np.array(selected_x)
selected_y = np.array(selected_y)
print(len(selected_x))
print(len(X))

x_train = selected_x
y_train = selected_y

import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


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
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
