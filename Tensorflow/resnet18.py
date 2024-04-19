import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, \
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import scipy.misc
from matplotlib.pyplot import imshow
from contextlib import redirect_stdout

# IDENTITY BLOCK

# x is input, y=F(x)
# identity block simply means input should be equal to output.
#  y = x + F(x)   the layers in a traditional network are learning the true output H(x)
# F(x) = y - x   the layers in a residual network are learning the residual F(x)
# Hence, the name: Residual Block.


def identity_block(X, f, filters, stage, block):
    """

    Arguments:
    X -- input of shape (m, height, width, channel)
    f -- shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters            # F1,F2 = number of filters   & f = filter size

    # Saving the input value.we need this later to add to the output.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    # X = Activation('relu')(X)


    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# CONVOLUTIONAL BLOCK

def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value
    X_shortcut = X

    # First layer
    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(s, s), padding='same', name=conv_name_base + '2a', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)  # 1,1 is filter size
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)  # normalization on channels
    X = Activation('relu')(X)

    # Second layer  (f,f)=3*3 filter by default
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    # X = Activation('relu')(X)


    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(s, s), padding='same',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0), name=conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value here, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# CREATING ResNet 18

# Each ResNet block is either 2 layer deep
def ResNet18(input_shape=(128, 384, 441), classes=2):
# def ResNet18(input_shape=(128, 384, 3), classes=2):
    """
    Implementation of the ResNet50 architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK -> CONVBLOCK -> IDBLOCK
    -> CONVBLOCK -> IDBLOCK -> CONVBLOCK -> IDBLOCK -> AVGPOOL -> TOPLAYER

    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)  # 3,3 padding

    # Initial Stage
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)  # 64 filters of 7*7
    X = BatchNormalization(axis=3, name='bn_conv1')(X)  # batchnorm applied on channels
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)  # window size is 3*3

    # Stage 1
    X = convolutional_block(X, f=3, filters=[64, 64], stage=1, block='a', s=1)
    X = identity_block(X, 3, [64, 64], stage=1, block='b')

    # Stage 2
    X = convolutional_block(X, f=3, filters=[128, 128], stage=2, block='a', s=2)
    X = identity_block(X, 3, [128, 128], stage=2, block='b')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[256, 256], stage=3, block='a', s=2)
    X = identity_block(X, 3, [256, 256], stage=3, block='b')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[512, 512], stage=4, block='a', s=2)
    X = identity_block(X, 3, [512, 512], stage=4, block='b')

    # AVGPOOL
    X = AveragePooling2D((2, 2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc' + str(classes),
              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet18')

    return model

model = ResNet18(input_shape = (128, 384, 441), classes = 2)
# model = ResNet18(input_shape = (128, 384, 3), classes = 2)

model.summary()

# Specify the path where you want to save the summary
summary_path = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/model_summary.txt'

# Open the file in write mode and redirect the summary to the file  
with open(summary_path, 'w') as f:
    with redirect_stdout(f):
        model.summary()