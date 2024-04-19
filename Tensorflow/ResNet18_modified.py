import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model

from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

def resnet_block(inputs, num_filters, kernel_size, strides, activation='relu'):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer = glorot_uniform(seed=0))(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer = glorot_uniform(seed=0))(x)
    x = BatchNormalization()(x)

    if strides > 1:
        shortcut = Conv2D(num_filters, kernel_size=1, strides=strides, padding='same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer = glorot_uniform(seed=0))(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.add([x, shortcut])
    x = Activation(activation)(x)
    return x

def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = resnet_block(x, num_filters=64, kernel_size=3, strides=1)
    x = resnet_block(x, num_filters=64, kernel_size=3, strides=1)

    x = resnet_block(x, num_filters=128, kernel_size=3, strides=2)
    x = resnet_block(x, num_filters=128, kernel_size=3, strides=1)

    x = resnet_block(x, num_filters=256, kernel_size=3, strides=2)
    x = resnet_block(x, num_filters=256, kernel_size=3, strides=1)

    x = resnet_block(x, num_filters=512, kernel_size=3, strides=2)
    x = resnet_block(x, num_filters=512, kernel_size=3, strides=1)

    x = AveragePooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name="avg_pool")(x)  # softmax is for multi_class and sigmoid is for binary class

    model = Model(inputs=inputs, outputs=x)
    return model

# Define the input shape and number of classes
input_shape = (128, 384, 441)
num_classes = 1

# Build the ResNet-18 model
model = build_resnet(input_shape, num_classes)

# Print the model summary
model.summary()