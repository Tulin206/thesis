import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential

def SpectroCNN(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid'))

    return model

input_shape = (128, 384, 441)  # IR spectroscopic data shape
num_classes = 1

model = SpectroCNN(input_shape, num_classes)
model.summary()