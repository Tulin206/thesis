import os
import scipy
from scipy.io import loadmat
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from glob import glob



# Transfer data from matlab file into Python
test_data_I1 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_I_Sample_1_neu_korr.mat")
#print("type:", type(test_data_I1))
#print("len:", len(test_data_I1))
#print("keys:", test_data_I1.keys())



test_I1 = test_data_I1["spek"]
print("\n\n")
print("Shape of spek:", test_I1.shape)
# print("\n\n")
# print("print before scaling\n\n", test_I1)
# print("\n\n")

# Transpose the input image
test_I1_tr = np.transpose(test_I1)
print("Shape of spek after being transposed:", test_I1_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_I1_tr = scaler.fit_transform(test_I1_tr)
print("Shape of spek after being scaled:", test_I1_tr.shape)
# print("print after scaling\n\n", test_I1_tr)
# print("\n\n")


# Reshape the image
test_dataset_1 = np.reshape(test_I1_tr, (128, 384, 441))
#test_dataset_1 = np.reshape(img_plot, (4, 12, 1024))
np_darray_1 = np.array(test_dataset_1)
print('After reshaping the image dimension', np_darray_1.shape)
# save the array as an .npy file
testdataset1 = np.save('testdataset_1.npy', np_darray_1)

# Visualise the original image
img_sum = np.sum(test_I1, axis=0)
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



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_I2 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_I_Sample_2_korr.mat")
# print("type:", type(test_data_I2))
# print("len:", len(test_data_I2))
# print("keys:", test_data_I2.keys())

test_I2 = test_data_I2["spek"]
print("Shape of spek:", test_I2.shape)

# Transpose the input image
test_I2_tr = np.transpose(test_I2)
print("Shape of spek after being transposed:", test_I2_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_I2_tr = scaler.fit_transform(test_I2_tr)
print("Shape of spek after being scaled:", test_I2_tr.shape)


# Reshape the image
test_dataset_2 = np.reshape(test_I2_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_2 = np.array(test_dataset_2)
print('After reshaping the image dimension', np_darray_2.shape)
# save the array as an .npy file
testdataset2 = np.save('testdataset_2.npy', np_darray_2)

# Visualise the original image
img_sum = np.sum(test_I2, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
test_data_II8 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_II_Sample_8_neu_korr.mat")
#print(test_data_II8)
# print("type:", type(test_data_II8))
# print("len:", len(test_data_II8))
# print("keys:", test_data_II8.keys())

test_II8 = test_data_II8["spek"]
print("Shape of spek:", test_II8.shape)

# Transpose the input image
test_II8_tr = np.transpose(test_II8)
print("Shape of spek after being transposed:", test_II8_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_II8_tr = scaler.fit_transform(test_II8_tr)
print("Shape of spek after being scaled:", test_II8_tr.shape)


# Reshape the image
test_dataset_3 = np.reshape(test_II8_tr, (128, 384, 441))
#dataset_3 = np.reshape(img_plot, (4, 12, 1024))
np_darray_3 = np.array(test_dataset_3)
print('After reshaping the image dimension', np_darray_3.shape)
# save the array as an .npy file
testdataset3 = np.save('testdataset_3.npy', np_darray_3)

# Visualise the original image
img_sum = np.sum(test_II8, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()


# Transfer data from matlab file into Python
test_data_III10 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_III_Sample_10_korr.mat")
# print("type:", type(test_data_III10))
# print("len:", len(test_data_III10))
# print("keys:", test_data_III10.keys())

test_III10 = test_data_III10["spek"]
print("Shape of spek:", test_III10.shape)

# Transpose the input image
test_III10_tr = np.transpose(test_I1)
print("Shape of spek after being transposed:", test_III10_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_III10_tr = scaler.fit_transform(test_III10_tr)
print("Shape of spek after being scaled:", test_III10_tr.shape)


# reshape the image
test_dataset_4 = np.reshape(test_III10_tr, (128, 384, 441))
#dataset_4 = np.reshape(img_plot, (4, 12, 1024))
np_darray_4 = np.array(test_dataset_4)
print('After reshaping the image dimension', np_darray_4.shape)
# save the array as an .npy file
testdataset4 = np.save('testdataset_4.npy', np_darray_4)
#plt.imshow(img_plot.T, cmap ="jet")


# Visualise the original image
img_sum = np.sum(test_III10, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()


# Transfer data from matlab file into Python
test_data_v17 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_V_Sample_17_korr.mat")
# print("type:", type(test_v17))
# print("len:", len(test_v17))
# print("keys:", test_v17.keys())

test_v17 = test_data_v17["spek"]
print("Shape of spek:", test_v17.shape)

# Transpose the input image
test_v17_tr = np.transpose(test_v17)
print("Shape of spek after being transposed:", test_v17_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_v17_tr = scaler.fit_transform(test_v17_tr)
print("Shape of spek after being scaled:", test_v17_tr.shape)


# Reshape the image
test_dataset_5 = np.reshape(test_v17_tr, (128, 384, 441))
#dataset_5 = np.reshape(img_plot, (4, 12, 1024))
np_darray_5 = np.array(test_dataset_5)
print('After reshaping the image dimension', np_darray_5.shape)
# save the array as an .npy file
testdataset5 = np.save('testdataset_5.npy', np_darray_5)
#print(dataset_5)

# Visualise the original image
img_sum = np.sum(test_v17, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_I3 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_I_Sample_3_neu2.mat")
# print("type:", type(test_data_I2))
# print("len:", len(test_data_I2))
# print("keys:", test_data_I2.keys())

test_I3 = test_data_I3["spek"]
print("Shape of spek:", test_I3.shape)

# Transpose the input image
test_I3_tr = np.transpose(test_I3)
print("Shape of spek after being transposed:", test_I3_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_I3_tr = scaler.fit_transform(test_I3_tr)
print("Shape of spek after being scaled:", test_I3_tr.shape)


# reshape the image
test_dataset_6 = np.reshape(test_I3_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_6 = np.array(test_dataset_6)
print('After reshaping the image dimension', np_darray_6.shape)
# save the array as an .npy file
testdataset6 = np.save('testdataset_6.npy', np_darray_6)

# Visualise the original image
img_sum = np.sum(test_I3, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()


# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_III11 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_III_Sample_11.mat")
# print("type:", type(test_data_III11))
# print("len:", len(test_data_III11))
# print("keys:", test_data_III11.keys())

test_III11 = test_data_III11["spek"]
print("Shape of spek:", test_III11.shape)

# Transpose the input image
test_III11_tr = np.transpose(test_III11)
print("Shape of spek after being transposed:", test_III11_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_III11_tr = scaler.fit_transform(test_III11_tr)
print("Shape of spek after being scaled:", test_III11_tr.shape)


# Reshape the image
test_dataset_7 = np.reshape(test_III11_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_7 = np.array(test_dataset_7)
print('After reshaping the image dimension', np_darray_7.shape)
# save the array as an .npy file
testdataset7 = np.save('testdataset_7.npy', np_darray_7)


# Visualise the original image
img_sum = np.sum(test_III11, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()


# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_IV13 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_IV_Sample_13.mat")
# print("type:", type(test_data_IV13))
# print("len:", len(test_data_IV13))
# print("keys:", test_data_IV13.keys())

test_IV13 = test_data_IV13["spek"]
print("Shape of spek:", test_IV13.shape)

# Transpose the input image
test_IV13_tr = np.transpose(test_IV13)
print("Shape of spek after being transposed:", test_IV13_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_IV13_tr = scaler.fit_transform(test_IV13_tr)
print("Shape of spek after being scaled:", test_IV13_tr.shape)


# Reshape the input image
test_dataset_8 = np.reshape(test_IV13_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_8 = np.array(test_dataset_8)
print('After reshaping the image dimension', np_darray_8.shape)
# save the array as an .npy file
testdataset8 = np.save('testdataset_8.npy', np_darray_8)


# Visualise the original image
img_sum = np.sum(test_IV13, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_IV14 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_IV_Sample_14.mat")
# print("type:", type(test_data_IV14))
# print("len:", len(test_data_IV14))
# print("keys:", test_data_IV14.keys())

test_IV14 = test_data_IV14["spek"]
print("Shape of spek:", test_IV14.shape)

# Transpose the input image
test_IV14_tr = np.transpose(test_IV14)
print("Shape of spek after being transposed:", test_IV14_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_IV14_tr = scaler.fit_transform(test_IV14_tr)
print("Shape of spek after being scaled:", test_IV14_tr.shape)


# Reshape the image
test_dataset_9 = np.reshape(test_IV14_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_9 = np.array(test_dataset_9)
print('After reshaping the image dimension', np_darray_9.shape)
# save the array as an .npy file
testdataset9 = np.save('testdataset_9.npy', np_darray_9)


# Visualise the original image
img_sum = np.sum(test_IV14, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_4 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_4_neuII.mat")
# print("type:", type(test_data_4))
# print("len:", len(test_data_4))
# print("keys:", test_data_4.keys())

test_4 = test_data_4["spek"]
print("Shape of spek:", test_4.shape)

# Transpose the input image
test_4_tr = np.transpose(test_4)
print("Shape of spek after being transposed:", test_4_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_4_tr = scaler.fit_transform(test_4_tr)
print("Shape of spek after being scaled:", test_4_tr.shape)


# Reshape the input image
test_dataset_10 = np.reshape(test_4_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_10 = np.array(test_dataset_10)
print('After reshaping the image dimension', np_darray_10.shape)
# save the array as an .npy file
testdataset10 = np.save('testdataset_10.npy', np_darray_10)


# Visualise the original image
img_sum = np.sum(test_4, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_6 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_6_neuII.mat")
# print("type:", type(test_data_6))
# print("len:", len(test_data_6))
# print("keys:", test_data_6.keys())

test_6 = test_data_6["spek"]
print("Shape of spek:", test_6.shape)

# Transpose the input image
test_6_tr = np.transpose(test_6)
print("Shape of spek after being transposed:", test_6_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_6_tr = scaler.fit_transform(test_6_tr)
print("Shape of spek after being scaled:", test_6_tr.shape)


# Reshape the input image
test_dataset_11 = np.reshape(test_6_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_11 = np.array(test_dataset_11)
print('After reshaping the image dimension', np_darray_11.shape)
# save the array as an .npy file
testdataset11 = np.save('testdataset_11.npy', np_darray_11)


# Visualise the original image
img_sum = np.sum(test_6, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_7 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_7_neu.mat")
# print("type:", type(test_data_7))
# print("len:", len(test_data_7))
# print("keys:", test_data_7.keys())

test_7 = test_data_7["spek"]
print("Shape of spek:", test_7.shape)

# Transpose the input image
test_7_tr = np.transpose(test_7)
print("Shape of spek after being transposed:", test_7_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_7_tr = scaler.fit_transform(test_7_tr)
print("Shape of spek after being scaled:", test_7_tr.shape)


# Reshape the input image
test_dataset_12 = np.reshape(test_7_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_12 = np.array(test_dataset_12)
print('After reshaping the image dimension', np_darray_12.shape)
# save the array as an .npy file
testdataset12 = np.save('testdataset_12.npy', np_darray_12)


# Visualise the original image
img_sum = np.sum(test_7, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_9 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_9_neuII.mat")
# print("type:", type(test_data_9))
# print("len:", len(test_data_9))
# print("keys:", test_data_9.keys())

test_9 = test_data_9["spek"]
print("Shape of spek:", test_9.shape)

# Transpose the input image
test_9_tr = np.transpose(test_9)
print("Shape of spek after being transposed:", test_9_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_9_tr = scaler.fit_transform(test_9_tr)
print("Shape of spek after being scaled:", test_9_tr.shape)


# Reshape the image
test_dataset_13 = np.reshape(test_9_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_13 = np.array(test_dataset_13)
print('After reshaping the image dimension', np_darray_13.shape)
# save the array as an .npy file
testdataset13 = np.save('testdataset_13.npy', np_darray_13)


# Visualise the original image
img_sum = np.sum(test_9, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_15 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_15_neu.mat")
# print("type:", type(test_data_15))
# print("len:", len(test_data_15))
# print("keys:", test_data_15.keys())

test_15 = test_data_15["spek"]
print("Shape of spek:", test_15.shape)

# Transpose the input image
test_15_tr = np.transpose(test_15)
print("Shape of spek after being transposed:", test_15_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_15_tr = scaler.fit_transform(test_15_tr)
print("Shape of spek after being scaled:", test_I1_tr.shape)


# Reshape the input image
test_dataset_14 = np.reshape(test_15_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_14 = np.array(test_dataset_14)
print('After reshaping the image dimension', np_darray_14.shape)
# save the array as an .npy file
testdataset14 = np.save('testdataset_14.npy', np_darray_14)


# Visualise the image
img_sum = np.sum(test_15, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_16 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_16_neu.mat")
# print("type:", type(test_data_16))
# print("len:", len(test_data_16))
# print("keys:", test_data_16.keys())

test_16 = test_data_16["spek"]
print("Shape of spek:", test_16.shape)

# Transpose the input image
test_16_tr = np.transpose(test_16)
print("Shape of spek after being transposed:", test_16_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_16_tr = scaler.fit_transform(test_16_tr)
print("Shape of spek after being scaled:", test_16_tr.shape)


# Reshape the input image
test_dataset_15 = np.reshape(test_16_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_15 = np.array(test_dataset_15)
print('After reshaping the image dimension', np_darray_15.shape)
# save the array as an .npy file
testdataset15 = np.save('testdataset_15.npy', np_darray_15)


# Visualise the original image
img_sum = np.sum(test_16, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_18 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_18_neu.mat")
# print("type:", type(test_data_18))
# print("len:", len(test_data_18))
# print("keys:", test_data_18.keys())

test_18 = test_data_18["spek"]
print("Shape of spek:", test_18.shape)

# Transpose the input image
test_18_tr = np.transpose(test_18)
print("Shape of spek after being transposed:", test_18_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_18_tr = scaler.fit_transform(test_18_tr)
print("Shape of spek after being scaled:", test_18_tr.shape)


# Reshape the input image
test_dataset_16 = np.reshape(test_18_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_16 = np.array(test_dataset_16)
print('After reshaping the image dimension', np_darray_16.shape)
# save the array as an .npy file
testdataset16 = np.save('testdataset_16.npy', np_darray_16)


# Visualise the original image
img_sum = np.sum(test_18, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_19 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_19_neu.mat")
# print("type:", type(test_data_19))
# print("len:", len(test_data_19))
# print("keys:", test_data_19.keys())

test_19 = test_data_19["spek"]
print("Shape of spek:", test_19.shape)

# Transpose the input image
test_19_tr = np.transpose(test_19)
print("Shape of spek after being transposed:", test_19_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_19_tr = scaler.fit_transform(test_19_tr)
print("Shape of spek after being scaled:", test_I1_tr.shape)


# Reshape the input image
test_dataset_17 = np.reshape(test_19_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_17 = np.array(test_dataset_17)
print('After reshaping the image dimension', np_darray_17.shape)
# save the array as an .npy file
testdataset17 = np.save('testdataset_17.npy', np_darray_17)


# Visualise the original image
img_sum = np.sum(test_19, axis=0)
print('summation of 441 wavelength for speak', img_sum.shape)
img_plot = np.reshape(img_sum, (128,384))
print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
plt.imshow(img_plot, cmap ="jet")
print('////////////////////////////////////////////////////////////\n')
#plt.show()



# Transfer data from matlab file into Python
from scipy.io import loadmat
test_data_21 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_21_neu.mat")
# print("type:", type(test_data_21))
# print("len:", len(test_data_21))
# print("keys:", test_data_21.keys())

test_21 = test_data_21["spek"]
print("Shape of spek:", test_21.shape)

# Transpose the input image
test_21_tr = np.transpose(test_21)
print("Shape of spek after being transposed:", test_21_tr.shape)

# define standard scaler
scaler = StandardScaler()

# transform data
test_21_tr = scaler.fit_transform(test_21_tr)
print("Shape of spek after being scaled:", test_21_tr.shape)


# Reshape the input image
test_dataset_18 = np.reshape(test_21_tr, (128, 384, 441))
#test_dataset_2 = np.reshape(img_plot, (4, 12, 1024))
np_darray_18 = np.array(test_dataset_18)
print('After reshaping the image dimension', np_darray_18.shape)
# save the array as an .npy file
testdataset18 = np.save('testdataset_18.npy', np_darray_18)


# Visualise the original image
img_sum = np.sum(test_21, axis=0)
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