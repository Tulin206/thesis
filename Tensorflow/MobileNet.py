import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, AvgPool2D, GlobalAveragePooling2D,\
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, ReLU, DepthwiseConv2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import scipy.misc
from matplotlib.pyplot import imshow

# MobileNet block

def mobilnet_block (x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


# stem of the model
input = Input(shape = (128, 384, 441))
num_classes = 1

x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)

# main part of the modelx = mobilnet_block(x, filters = 64, strides = 1)
x = mobilnet_block(x, filters=64, strides=1)
x = mobilnet_block(x, filters=128, strides=2)
x = mobilnet_block(x, filters=128, strides=1)
x = mobilnet_block(x, filters=256, strides=2)
x = mobilnet_block(x, filters=256, strides=1)
x = mobilnet_block(x, filters=512, strides=2)

for _ in range(5):
    x = mobilnet_block(x, filters=512, strides=1)

x = mobilnet_block(x, filters=1024, strides=2)
x = mobilnet_block(x, filters=1024, strides=1)

# Global average pooling and output layer
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes, activation='sigmoid')(x)

# x = AveragePooling2D(pool_size=(7, 7), strides=1, data_format='channels_last')(x)
# output = Dense(units=2, activation='softmax')(x)
model = Model(inputs=input, outputs=outputs)
model.summary()

# plot the model
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=False,show_layer_names=True,
                          rankdir='TB', expand_nested=False, dpi=96)