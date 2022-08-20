import ssl

import numpy as np
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from tensorflow import keras

ssl._create_default_https_context = ssl._create_unverified_context

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

classifier = Sequential()
classifier.add(
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Flatten())
classifier.add(Dense(10, activation='softmax'))

classifier.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


classifier.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)


acc = classifier.evaluate(x_test, y_test, verbose=0)
print('accuracy = %.3f' % (acc * 100.0))
