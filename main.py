# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import scipy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import tensorflow as tf
import keras
from keras import datasets, layers, models
from keras import initializers
from keras.models import Sequential, load_model, Model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.utils import array_to_img, img_to_array, load_img
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from IPython.display import SVG
# from tensorflow.keras.utils.vis_utils import model_to_dot
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob


from scipy.io import loadmat
annots1 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_I_Sample_1_neu_korr.mat")
#print("type:", type(annots1))
#print("len:", len(annots1))
#print("keys:", annots1.keys())


from scipy.io import loadmat
annots2 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_I_Sample_2_korr.mat")
# print("type:", type(annots2))
# print("len:", len(annots2))
# print("keys:", annots2.keys())


annots3 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_II_Sample_8_neu_korr.mat")
#print(annots2)
# print("type:", type(annots3))
# print("len:", len(annots3))
# print("keys:", annots3.keys())

annots4 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_III_Sample_10_korr.mat")
# print("type:", type(annots4))
# print("len:", len(annots4))
# print("keys:", annots4.keys())

annots5 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_V_Sample_17_korr.mat")
# print("type:", type(annots5))
# print("len:", len(annots5))
# print("keys:", annots5.keys())

# Transfer data from matlab file into Python
training20 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_20_neu.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

t20 = training20["spek"]
print("Shape of spek:", t20.shape)
training_set_1 = np.reshape(t20, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_1 = np.array(training_set_1)
print('After reshaping the image dimension', np_tarray_1.shape)
# save the array as an .npy file
training_set1 = np.save('trainingset_1.npy', np_tarray_1)
img_sum = np.sum(t20, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training22 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_22_neu.mat")
print("type:", type(training22))
print("len:", len(training22))
print("keys:", training22.keys())
print('//////////////////\n')

t22 = training22["spek"]
print("Shape of spek:", t22.shape)
training_set_2 = np.reshape(t22, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_2 = np.array(training_set_2)
print('After reshaping the image dimension', np_tarray_2.shape)
# save the array as an .npy file
training_set2 = np.save('trainingset_2.npy', np_tarray_2)
img_sum = np.sum(t22, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training23 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_23_neu.mat")
print("type:", type(training23))
print("len:", len(training23))
print("keys:", training23.keys())
print('//////////////////\n')

t23 = training23["spek"]
print("Shape of spek:", t23.shape)
training_set_3 = np.reshape(t23, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_3 = np.array(training_set_3)
print('After reshaping the image dimension', np_tarray_3.shape)
# save the array as an .npy file
training_set3 = np.save('trainingset_3.npy', np_tarray_3)
img_sum = np.sum(t23, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training25 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_25_neu.mat")
print("type:", type(training25))
print("len:", len(training25))
print("keys:", training25.keys())
print('//////////////////\n')

t25 = training25["spek"]
print("Shape of spek:", t25.shape)
training_set_4 = np.reshape(t25, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_4 = np.array(training_set_4)
print('After reshaping the image dimension', np_tarray_4.shape)
# save the array as an .npy file
training_set4 = np.save('trainingset_4.npy', np_tarray_4)
img_sum = np.sum(t25, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training28 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_28_neu2.mat")
print("type:", type(training28))
print("len:", len(training28))
print("keys:", training28.keys())
print('//////////////////\n')

t28 = training28["spek"]
print("Shape of spek:", t28.shape)
training_set_5 = np.reshape(t28, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_5 = np.array(training_set_5)
print('After reshaping the image dimension', np_tarray_5.shape)
# save the array as an .npy file
training_set5 = np.save('trainingset_5.npy', np_tarray_5)
img_sum = np.sum(t28, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training29 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_29_neu2.mat")
print("type:", type(training29))
print("len:", len(training29))
print("keys:", training29.keys())
print('//////////////////\n')

t29 = training29["spek"]
print("Shape of spek:", t29.shape)
training_set_6 = np.reshape(t29, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_6 = np.array(training_set_6)
print('After reshaping the image dimension', np_tarray_6.shape)
# save the array as an .npy file
training_set6 = np.save('trainingset_6.npy', np_tarray_6)
img_sum = np.sum(t29, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training31 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_31_neu.mat")
print("type:", type(training31))
print("len:", len(training31))
print("keys:", training31.keys())
print('//////////////////\n')

t31 = training31["spek"]
print("Shape of spek:", t31.shape)
training_set_7 = np.reshape(t31, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_7 = np.array(training_set_7)
print('After reshaping the image dimension', np_tarray_7.shape)
# save the array as an .npy file
training_set7 = np.save('trainingset_7.npy', np_tarray_7)
img_sum = np.sum(t31, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training32 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_32_neu.mat")
print("type:", type(training32))
print("len:", len(training32))
print("keys:", training32.keys())
print('//////////////////\n')

t32 = training32["spek"]
print("Shape of spek:", t32.shape)
training_set_8 = np.reshape(t32, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_8 = np.array(training_set_8)
print('After reshaping the image dimension', np_tarray_8.shape)
# save the array as an .npy file
training_set8 = np.save('trainingset_8.npy', np_tarray_8)
img_sum = np.sum(t32, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training33 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_33_neuII.mat")
print("type:", type(training33))
print("len:", len(training33))
print("keys:", training33.keys())
print('//////////////////\n')

t33 = training33["spek"]
print("Shape of spek:", t33.shape)
training_set_9 = np.reshape(t33, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_9 = np.array(training_set_9)
print('After reshaping the image dimension', np_tarray_9.shape)
# save the array as an .npy file
training_set9 = np.save('trainingset_9.npy', np_tarray_9)
img_sum = np.sum(t33, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

training34 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_34_neuII.mat")
print("type:", type(training34))
print("len:", len(training34))
print("keys:", training34.keys())
print('//////////////////\n')

t34 = training34["spek"]
print("Shape of spek:", t34.shape)
training_set_10 = np.reshape(t34, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_10 = np.array(training_set_10)
print('After reshaping the image dimension', np_tarray_10.shape)
# save the array as an .npy file
training_set10 = np.save('trainingset_10.npy', np_tarray_10)
img_sum = np.sum(t34, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')



x10 = annots1["spek"]
print("Shape of spek:", x10.shape)
dataset_1 = np.reshape(x10, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_array_1 = np.array(dataset_1)
print('After reshaping the image dimension', np_array_1.shape)
# save the array as an .npy file
dataset1 = np.save('dataset_1.npy', np_array_1)
img_sum = np.sum(x10, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()
#plt.imshow(img_plot, cmap="hot")
#plt.imshow(img_plot, cmap= matplotlib.cm.get_cmap('Spectral'))
#plt.specgram(img_plot,NFFT = 1024, Fs = 6, noverlap =
# print("First image of spek:", x10[0])
# print("Shape of First image of spek:", x10[0].shape)
# print(x10)

#image = x[0].reshape(3, 128, 128)
#print((plt.imshow(x[0])))


y10 = annots2["spek"]
print("Shape of spek:", y10.shape)
dataset_2 = np.reshape(y10, (128, 384, 441))
#dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_array_2 = np.array(dataset_2)
print('After reshaping the image dimension', np_array_2.shape)
# save the array as an .npy file
dataset2 = np.save('dataset_2.npy', np_array_2)
img_sum = np.sum(y10, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()

z10 = annots3["spek"]
print("Shape of spek:", z10.shape)
dataset_3 = np.reshape(z10, (128, 384, 441))
#dataset_3 = np.reshape(img_plot, (4, 12, 1024))
np_array_3 = np.array(dataset_3)
print('After reshaping the image dimension', np_array_3.shape)
# save the array as an .npy file
dataset3 = np.save('dataset_3.npy', np_array_3)
img_sum = np.sum(z10, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()

a10 = annots4["spek"]
print("Shape of spek:", a10.shape)
dataset_4 = np.reshape(a10, (128, 384, 441))
#dataset_4 = np.reshape(img_plot, (4, 12, 1024))
np_array_4 = np.array(dataset_4)
print('After reshaping the image dimension', np_array_4.shape)
# save the array as an .npy file
dataset4 = np.save('dataset_4.npy', np_array_4)
#plt.imshow(img_plot.T, cmap ="jet")
img_sum = np.sum(a10, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()

b10 = annots5["spek"]
print("Shape of spek:", b10.shape)
dataset_5 = np.reshape(b10, (128, 384, 441))
#dataset_5 = np.reshape(img_plot, (4, 12, 1024))
np_array_5 = np.array(dataset_5)
print('After reshaping the image dimension', np_array_5.shape)
# save the array as an .npy file
dataset5 = np.save('dataset_5.npy', np_array_5)
#print(dataset_5)
img_sum = np.sum(b10, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()


# x1 = annots1["Binning"]
# print("Shape of Binning:", x1.shape)
# print("First image of Binning:", x1[0])
# print("Shape of First image of Binning:", x1[0].shape)
# print(x1)
#
# x2 = annots1["Spek"]
# print("Shape of Spek:", x2.shape)
# print("First image of Spek:", x2[0])
# print("Shape of First image of Spek:", x2[0].shape)
# print(x2)
#
# x3 = annots1["a"]
# print("Shape of a:", x3.shape)
# print("First image of a:", x3[0])
# print("Shape of First image of a:", x3[0].shape)
# print(x3)
#
# x4 = annots1["ans"]
# print("Shape of ans:", x4.shape)
# print("First image of ans:", x4[0])
# print("Shape of First image of ans:", x4[0].shape)
# print(x4)
#
# x5 = annots1["bb"]
# print("Shape of bb:", x5.shape)
# #print("First image of ans:", x4[0])
# #print("Shape of First image of ans:", x4[0].shape)
# print(x5)
#
# x6 = annots1["dx"]
# print("Shape of dx:", x6.shape)
# #print("First image of ans:", x4[0])
# #print("Shape of First image of ans:", x4[0].shape)
# print(x6)
#
# x7 = annots1["dy"]
# print("Shape of dy:", x7.shape)
# #print("First image of ans:", x4[0])
# #print("Shape of First image of ans:", x4[0].shape)
# print(x7)
#
# x8 = annots1["pos"]
# print("Shape of pos:", x8.shape)
# #print("First image of ans:", x4[0])
# #print("Shape of First image of ans:", x4[0].shape)
# print(x8)
#
# x9 = annots1["s2"]
# print("Shape of s2:", x9.shape)
# #print("First image of ans:", x4[0])
# #print("Shape of First image of ans:", x4[0].shape)
# print(x9)
#
#
# x0 = annots1["x"]
# print("Shape of x:", x0.shape)
# print("First image of x:", x0[0])
# print("Shape of First image of x:", x0[0].shape)
# print(x0)

# Creating Image Size of Spectral Data
IMAGE_SIZE = [128, 384]

# Load the 5 spectral datasets into a list
datasets = [np.load('dataset_1.npy'), np.load('dataset_2.npy'), np.load('dataset_3.npy'), np.load('dataset_4.npy'), np.load('dataset_5.npy')]
print('Shape of whole datasets', np.shape(datasets))
x = np.array(datasets)
#data = np.vstack([dataset1, dataset2, dataset3, dataset4, dataset5])

# Generate labels for each image
classes = np.array([1, 1, 1, 0, 1]) # 0 represents one class, 1 represents the other

# Split the datasets and labels into training and testing sets
X_train, y_train = x[0:], classes[0:]

#Dimension of the Spectral dataset
print('Shape of training dataset and label:')
print((np.shape(X_train),y_train.shape))


# Load data
#train_path = '/mnt/ceph/tco/TCO-Students/Projects/Israt_Tulin/TrainingData/'
#train_path = '/mnt/ceph/tco/TCO-Students/Projects/Israt_Tulin/Training_Data/'

# Load Labels
#folders = glob('/mnt/ceph/tco/TCO-Students/Projects/Israt_Tulin/TrainingData/*')

# Print the labels
#print(folders)
#print(len(folders))

#Onehot Encoding the labels.
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical


#Since we have 5 classes we should expect the shape[1] of folders 1 to 10
#folders = to_categorical(len(folders) - 1)
#print(folders)


#Image Data Augmentation
train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, shear_range=0.2, zoom_range=0.2)

#Fitting the augmentation defined above to the data
train_aug = train_generator.flow(X_train, y_train, batch_size=32)
#train_augmented = train_generator.flow_from_directory(train_path, target_size=(128, 384), batch_size=1, class_mode="binary", shuffle=False, seed=42)
#print(train_aug.image_shape)
#print(train_augmented.image_shape)

#train_augmented = train_generator.flow_from_directory(train_path, folders, class_mode="categorical", shuffle=False, seed=42)
# img = load_img('/mnt/ceph/tco/TCO-Students/Projects/Israt_Tulin/TrainingData/Tumor_Grade_3_1_neu_korr/Figure_1.png')  # this is a PIL image
# x = img_to_array(img)
# x = x.reshape((1,) + x.shape)
#
# i = 0
# for batch in train_generator.flow(x, save_to_dir='/mnt/ceph/tco/TCO-Students/Projects/Israt_Tulin/Augmented_Data/', save_prefix='spectral', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break
#print (train_augmented)

# Creation of Identity Block for ResNet50
def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# Creation of convolutional Block for ResNet50
def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# ResNet50 model
#def ResNet50(input_shape=(128, 384, 3), classes=2):
def ResNet50(input_shape=(128, 384, 441), classes=2):
    global model
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    #X = X_input


    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL.
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    model.summary()

    return model

#model = ResNet50(input_shape = (128, 384, 3), classes = 2)
model = ResNet50(input_shape = (128, 384, 441), classes = 2)



# Compilation of resNet50 model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_aug, batch_size=1, epochs = 100)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
