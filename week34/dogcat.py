from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

# augments data on run by applying different transformations
training_generator = ImageDataGenerator(rescale=1. / 255,  # pixels go from 0-1 instead of 0-255
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

training_set = training_generator.flow_from_directory('dogcat_labelled/train',
                                                      target_size=(128, 128),  # resizes images
                                                      batch_size=32,  # each batch has 32 images
                                                      class_mode='binary')  # binary: 'dog' or 'cat'

testing_generator = ImageDataGenerator(rescale=1. / 255)  # pixels go from 0-1 instead of 0-255

test_set = testing_generator.flow_from_directory('dogcat_labelled/test',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(128, 128, 3)))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(MaxPooling2D((4, 4)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
classifier.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.01)
classifier.compile(optimizer=opt, loss='binary_crossentropy')

classifier.fit(training_set, steps_per_epoch=len(training_set), epochs=3, validation_data=test_set,
               validation_steps=len(test_set))

acc = classifier.evaluate(test_set, steps=len(test_set), verbose=0)
print('accuracy = %.3f' % (acc * 100.0))