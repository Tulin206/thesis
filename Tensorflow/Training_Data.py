import os
import scipy
from scipy.io import loadmat
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import array as arr
from array import *
import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from glob import glob


# Transfer data from matlab file into Python
training20 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_20_neu.mat")
# print("type:", type(training20))
# print("len:", len(training20))
# print("keys:", training20.keys())

t20 = training20["spek"]
print("Shape of spek:", t20.shape)

# Transpose the input image
t20_tr = np.transpose(t20)
print("Shape of spek after being transposed:", t20_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t20_tr = scaler.fit_transform(t20_tr)
print("Shape of spek after being scaled:", t20_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t20_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# Reshape input image
training_set_1 = np.reshape(t20_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_1 = np.array(training_set_1)
print('After reshaping the image dimension', np_tarray_1.shape)

# save the array as an .npy file
training_set1 = np.save('trainingset_1.npy', speak_pca)

# For visualize the original image
img_sum = np.sum(t20, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

datasets = np.load('trainingset_1.npy')
print(datasets.shape)

# Transfer data from matlab file into Python
training22 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_22_neu.mat")
# print("type:", type(training22))
# print("len:", len(training22))
# print("keys:", training22.keys())
# print('//////////////////\n')

t22 = training22["spek"]
print("Shape of spek:", t22.shape)

# Transpose the input image
t22_tr = np.transpose(t22)
print("Shape of spek after being transposed:", t22_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t22_tr = scaler.fit_transform(t22_tr)
print("Shape of spek after being scaled:", t22_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t22_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# Rehsape Input Image
training_set_2 = np.reshape(t22_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_2 = np.array(training_set_2)
print('After reshaping the image dimension', np_tarray_2.shape)
# save the array as an .npy file
training_set2 = np.save('trainingset_2.npy', speak_pca)

# For visualise the original image
img_sum = np.sum(t22, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

datasets = [np.load('trainingset_1.npy'), np.load('trainingset_2.npy')]
print(np.shape(datasets))

# Transfer data from matlab file into Python
training23 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_23_neu.mat")
# print("type:", type(training23))
# print("len:", len(training23))
# print("keys:", training23.keys())
# print('//////////////////\n')

t23 = training23["spek"]
print("Shape of spek:", t23.shape)

# Transpose the input image
t23_tr = np.transpose(t23)
print("Shape of spek after being transposed:", t23_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t23_tr = scaler.fit_transform(t23_tr)
print("Shape of spek after being scaled:", t23_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t23_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# Reshape input image
training_set_3 = np.reshape(t23_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_3 = np.array(training_set_3)
print('After reshaping the image dimension', np_tarray_3.shape)
# save the array as an .npy file
training_set3 = np.save('trainingset_3.npy', speak_pca)

# For visualise the original image
img_sum = np.sum(t23, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

datasets = [np.load('trainingset_1.npy'), np.load('trainingset_2.npy'), np.load('trainingset_3.npy')]
print(np.shape(datasets))

# Transfer data from matlab file into Python
training25 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_25_neu.mat")
# print("type:", type(training25))
# print("len:", len(training25))
# print("keys:", training25.keys())
# print('//////////////////\n')

t25 = training25["spek"]
print("Shape of spek:", t25.shape)

# Transpose the input image
t25_tr = np.transpose(t25)
print("Shape of spek after being transposed:", t25_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t25_tr = scaler.fit_transform(t25_tr)
print("Shape of spek after being scaled:", t20_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t25_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# Reshape input image
training_set_4 = np.reshape(t25_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_4 = np.array(training_set_4)
print('After reshaping the image dimension', np_tarray_4.shape)
# save the array as an .npy file
training_set4 = np.save('trainingset_4.npy', speak_pca)

# Visualise the original image
img_sum = np.sum(t25, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')


# Transfer data from matlab file into Python
training28 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_28_neu2.mat")
# print("type:", type(training28))
# print("len:", len(training28))
# print("keys:", training28.keys())
# print('//////////////////\n')

t28 = training28["spek"]
print("Shape of spek:", t28.shape)

# Transpose the input image
t28_tr = np.transpose(t28)
print("Shape of spek after being transposed:", t28_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t28_tr = scaler.fit_transform(t28_tr)
print("Shape of spek after being scaled:", t28_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t28_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# Reshape the image
training_set_5 = np.reshape(t28_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_5 = np.array(training_set_5)
print('After reshaping the image dimension', np_tarray_5.shape)
# save the array as an .npy file
training_set5 = np.save('trainingset_5.npy', speak_pca)


# Visualise the original image
img_sum = np.sum(t28, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')


# Transfer data from matlab file into Python
training29 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_29_neu2.mat")
# print("type:", type(training29))
# print("len:", len(training29))
# print("keys:", training29.keys())
# print('//////////////////\n')

t29 = training29["spek"]
print("Shape of spek:", t29.shape)

# Transpose the input image
t29_tr = np.transpose(t29)
print("Shape of spek after being transposed:", t29_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t29_tr = scaler.fit_transform(t29_tr)
print("Shape of spek after being scaled:", t29_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t29_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

#Reshape the image
training_set_6 = np.reshape(t29_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_6 = np.array(training_set_6)
print('After reshaping the image dimension', np_tarray_6.shape)
# save the array as an .npy file
training_set6 = np.save('trainingset_6.npy', speak_pca)


# Visualise the original image
img_sum = np.sum(t29, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')


# Transfer data from matlab file into Python
training31 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_31_neu.mat")
# print("type:", type(training31))
# print("len:", len(training31))
# print("keys:", training31.keys())
# print('//////////////////\n')

t31 = training31["spek"]
print("Shape of spek:", t31.shape)

# Transpose the input image
t31_tr = np.transpose(t31)
print("Shape of spek after being transposed:", t31_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t31_tr = scaler.fit_transform(t31_tr)
print("Shape of spek after being scaled:", t31_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t31_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# Reshape the image
training_set_7 = np.reshape(t31_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_7 = np.array(training_set_7)
print('After reshaping the image dimension', np_tarray_7.shape)
# save the array as an .npy file
training_set7 = np.save('trainingset_7.npy', speak_pca)


# Visualise the original image
img_sum = np.sum(t31, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')


# Transfer data from matlab file into Python
training32 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_32_neu.mat")
# print("type:", type(training32))
# print("len:", len(training32))
# print("keys:", training32.keys())
# print('//////////////////\n')

t32 = training32["spek"]
print("Shape of spek:", t32.shape)

# Transpose the input image
t32_tr = np.transpose(t32)
print("Shape of spek after being transposed:", t32_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t32_tr = scaler.fit_transform(t32_tr)
print("Shape of spek after being scaled:", t32_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t32_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# reshape the image
training_set_8 = np.reshape(t32_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_8 = np.array(training_set_8)
print('After reshaping the image dimension', np_tarray_8.shape)
# save the array as an .npy file
training_set8 = np.save('trainingset_8.npy', speak_pca)

# Visualise the original image
img_sum = np.sum(t32, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')


# Transfer data from matlab file into Python
training33 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_33_neuII.mat")
# print("type:", type(training33))
# print("len:", len(training33))
# print("keys:", training33.keys())
# print('//////////////////\n')

t33 = training33["spek"]
print("Shape of spek:", t33.shape)

# Transpose the input image
t33_tr = np.transpose(t33)
print("Shape of spek after being transposed:", t33_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t33_tr = scaler.fit_transform(t33_tr)
print("Shape of spek after being scaled:", t33_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t33_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# Reshape the image
training_set_9 = np.reshape(t33_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_9 = np.array(training_set_9)
print('After reshaping the image dimension', np_tarray_9.shape)
# save the array as an .npy file
training_set9 = np.save('trainingset_9.npy', speak_pca)


# Visualise the original image
img_sum = np.sum(t33, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')


# Transfer data from matlab file into Python
training34 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_34_neuII.mat")
# print("type:", type(training34))
# print("len:", len(training34))
# print("keys:", training34.keys())
# print('//////////////////\n')

t34 = training34["spek"]
print("Shape of spek:", t34.shape)

# Transpose the input image
t34_tr = np.transpose(t34)
print("Shape of spek after being transposed:", t34_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
t34_tr = scaler.fit_transform(t34_tr)
print("Shape of spek after being scaled:", t34_tr.shape)

pca = PCA(n_components=3)
speak_pca = pca.fit_transform(t34_tr)
print("After applying PCA, the sape of spek:", speak_pca.shape)
# print((np.reshape(speak_pca, (128, 384, 3))).shape)

# Reshape input image
speak_pca = speak_pca.reshape(128, 384, 3)
print('After reshaping the image dimension', speak_pca.shape)

# Reshape the image
training_set_10 = np.reshape(t34_tr, (128, 384, 441))
#dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_tarray_10 = np.array(training_set_10)
print('After reshaping the image dimension', np_tarray_10.shape)
# save the array as an .npy file
training_set10 = np.save('trainingset_10.npy', speak_pca)

# Visualise the original image
img_sum = np.sum(t34, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('//////////////////\n')

datasets = [np.load('trainingset_1.npy'), np.load('trainingset_2.npy'), np.load('trainingset_3.npy'),
            np.load('trainingset_4.npy'), np.load('trainingset_5.npy'), np.load('trainingset_6.npy'),
            np.load('trainingset_7.npy'), np.load('trainingset_8.npy'), np.load('trainingset_9.npy'), np.load('trainingset_10.npy')]

print('Shape of whole training datasets', np.shape(datasets))
print('\n')
print('End of printing the training datasets\n')