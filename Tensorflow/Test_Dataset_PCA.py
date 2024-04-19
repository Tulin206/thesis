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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from glob import glob

def load_test_data(array_of_speak,
                   dictionary_of_TrainingSet,
                   transposed_array_of_speak,
                   test_set_num,
                   NumpyArray_of_TestSet,
                   file_name):

    array_of_speak = dictionary_of_TrainingSet["spek"]
    print("Shape of spek:", array_of_speak.shape)

    # Calculate the variance along each channel (feature)
    channel_variances = np.var(array_of_speak, axis=0)

    # Sum the variances across all channels
    total_variance = np.sum(channel_variances)

    print("Total variance of the input image:", total_variance)

    # Transpose the input image
    transposed_array_of_speak = np.transpose(array_of_speak)
    print("Shape of spek after being transposed:", transposed_array_of_speak.shape)

    # Transpose the input image
    transposed_array_of_speak = np.transpose(array_of_speak)
    print("Shape of spek after being transposed:", transposed_array_of_speak.shape)

    # define standard scaler
    scaler = StandardScaler()

    # transform data
    transposed_array_of_speak = scaler.fit_transform(transposed_array_of_speak)
    print("Shape of spek after being scaled:", transposed_array_of_speak.shape)

    pca = PCA(n_components=3)
    speak_pca = pca.fit_transform(transposed_array_of_speak)
    print("After applying PCA, the shape of spek:", speak_pca.shape)
    # print((np.reshape(speak_pca, (128, 384, 3))).shape)

    # Accessing the explained variance of each component
    print('Explained variance of 3 components', pca.explained_variance_)
    print('Explained variance ratio of 3 components', pca.explained_variance_ratio_)
    print('Cumulative variance of 3 components', pca.explained_variance_.cumsum())

    # Reshape input image
    speak_pca = speak_pca.reshape(128, 384, 3)
    speak_pca = np.array(speak_pca)
    print('After reshaping the image dimension', speak_pca.shape)

    # # save the array as an .npy file
    # training_set_num = np.save('NumpyFile_of_TestSet.npy', NumpyArray_of_TestSet)

    def save_array_to_file(array_to_save: np.ndarray, file_name: str):
        np.save(file_name, array_to_save)

    save_array_to_file(speak_pca, file_name)

    # For visualize the original image
    img_sum = np.sum(array_of_speak, axis=0)
    print('summation of 441 wavelength for speak', img_sum.shape)
    img_plot = np.reshape(img_sum, (128, 384))
    print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
    plt.imshow(img_plot, cmap="jet")
    print('//////////////////\n')



# Transfer data from matlab file into Python
test_data_I2 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_I_Sample_2_korr.mat")

# Variable Declaration
test_I2 = []
test_I2_tr = []
test_dataset_2 = []
np_darray_2  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_1'


# function calling
load_test_data(test_I2, test_data_I2, test_I2_tr, test_dataset_2, np_darray_2 , str)


# Transfer data from matlab file into Python
test_data_II8 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_II_Sample_8_neu_korr.mat")

# Variable Declaration
test_II8 = []
test_II8_tr = []
test_dataset_3 = []
np_darray_3  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_2'


# function calling
load_test_data(test_II8, test_data_II8, test_II8_tr, test_dataset_3, np_darray_3 , str)


# Transfer data from matlab file into Python
test_data_IV13 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_IV_Sample_13.mat")


# Variable Declaration
test_IV13 = []
test_IV13_tr = []
test_dataset_8 = []
np_darray_8  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_3'


# function calling
load_test_data(test_IV13, test_data_IV13, test_IV13_tr, test_dataset_8, np_darray_8, str)


# Transfer data from matlab file into Python
test_data_IV14 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_IV_Sample_14.mat")

# Variable Declaration
test_IV14 = []
test_IV14_tr = []
test_dataset_9 = []
np_darray_9  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_4'


# function calling
load_test_data(test_IV14, test_data_IV14, test_IV14_tr, test_dataset_9, np_darray_9, str)


# Transfer data from matlab file into Python
test_data_4 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_4_neuII.mat")

# Variable Declaration
test_4 = []
test_4_tr = []
test_dataset_10 = []
np_darray_10  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_5'


# function calling
load_test_data(test_4, test_data_4, test_4_tr, test_dataset_10, np_darray_10, str)


# Transfer data from matlab file into Python
test_data_6 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_6_neuII.mat")

# Variable Declaration
test_6 = []
test_6_tr = []
test_dataset_11 = []
np_darray_11  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_6'


# function calling
load_test_data(test_6, test_data_6, test_6_tr, test_dataset_11, np_darray_11, str)


# Transfer data from matlab file into Python
test_data_9 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_9_neuII.mat")

# Variable Declaration
test_9 = []
test_9_tr = []
test_dataset_13 = []
np_darray_13  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_7'


# function calling
load_test_data(test_9, test_data_9, test_9_tr, test_dataset_13, np_darray_13, str)


# Transfer data from matlab file into Python
test_data_15 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_15_neu.mat")

# Variable Declaration
test_15 = []
test_15_tr = []
test_dataset_14 = []
np_darray_14  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_8'


# function calling
load_test_data(test_15, test_data_15, test_15_tr, test_dataset_14, np_darray_14, str)


# Transfer data from matlab file into Python
test_data_16 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_16_neu.mat")

# Variable Declaration
test_16 = []
test_16_tr = []
test_dataset_15 = []
np_darray_15  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_9'


# function calling
load_test_data(test_16, test_data_16, test_16_tr, test_dataset_15, np_darray_15, str)



# Transfer data from matlab file into Python
test_data_18 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_18_neu.mat")

# Variable Declaration
test_18 = []
test_18_tr = []
test_dataset_16 = []
np_darray_16  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_10'


# function calling
load_test_data(test_18, test_data_18, test_18_tr, test_dataset_16, np_darray_16, str)


# Transfer data from matlab file into Python
test_data_19 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_19_neu.mat")

# Variable Declaration
test_19 = []
test_19_tr = []
test_dataset_17 = []
np_darray_17  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_11'


# function calling
load_test_data(test_19, test_data_19, test_19_tr, test_dataset_17, np_darray_17, str)


# Transfer data from matlab file into Python
test_data_21 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_21_neu.mat")

# Variable Declaration
test_21 = []
test_21_tr = []
test_dataset_18 = []
np_darray_18  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_12'


# function calling
load_test_data(test_21, test_data_21, test_21_tr, test_dataset_18, np_darray_18, str)


# Load the 18 test spectral datasets into a list
testdatasets = [np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_1.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_2.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_3.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_4.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_5.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_6.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_7.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_8.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_9.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_10.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_11.npy'),
                np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_12.npy')]

print('Shape of whole test datasets', np.shape(testdatasets))
print('\n')
print('End of printing the test datasets\n')

