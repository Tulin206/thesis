import os
import scipy
from scipy.io import loadmat
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
from IPython.display import SVG
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from glob import glob

def load_test_data(array_of_speak,
                   dictionary_of_TrainingSet,
                   transposed_array_of_speak,
                   test_set_num,
                   NumpyArray_of_TestSet,
                   file_path):

    array_of_speak = dictionary_of_TrainingSet["spek"]
    print("Shape of spek:", array_of_speak.shape)

    # Transpose the input image
    transposed_array_of_speak = np.transpose(array_of_speak)
    print("Shape of spek after being transposed:", transposed_array_of_speak.shape)
    #
    # # define standard scaler
    # scaler = StandardScaler()
    #
    # # transform data
    # transposed_array_of_speak = scaler.fit_transform(array_of_speak)
    # print("Shape of spek after being scaled:", transposed_array_of_speak.shape)
    #
    # pca = PCA(n_components=3)
    # speak_pca = pca.fit_transform(transposed_array_of_speak)
    # print("After applying PCA, the shape of spek:", speak_pca.shape)
    #
    # # Reshape input image
    # speak_pca = np.reshape(speak_pca.T, (3, 128, 384))
    # speak_pca = np.array(speak_pca)
    # print('After reshaping the image dimension', speak_pca.shape)

    # # Reshape input image
    # test_set_num = np.sum(array_of_speak, axis=0)
    # test_set_num = np.reshape(test_set_num, (1, 128, 384))
    # # test_set_num = np.reshape(transposed_array_of_speak, (3, 128, 384))
    # # dataset_1 = np.reshape(img_plot, (4, 12, 1024))
    # NumpyArray_of_TestSet = np.array(test_set_num)
    # print('After reshaping the image dimension', NumpyArray_of_TestSet.shape)
    #
    # # save the array as an .npy file
    # # training_set_num = np.save('NumpyFile_of_TestSet.npy', NumpyArray_of_TestSet)

    def save_array_to_file(array_to_save: np.ndarray, file_path: str):
        np.save(file_path, array_to_save)

    save_array_to_file(transposed_array_of_speak, file_path)

    # # save_array_to_file(NumpyArray_of_TestSet, file_path)
    # save_array_to_file(test_set_num, file_path)

    # # For visualize the original image
    # # img_sum = np.sum(transposed_array_of_speak.T, axis=0)
    # # print('summation of 441 wavelength for speak', img_sum.shape)
    # # img_plot = np.reshape(img_sum, (128, 384))
    # img_plot = np.reshape(test_set_num, (128, 384))
    # print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
    # # plt.imshow(img_plot, cmap="jet")
    # # plt.show()

    # For visualize the original image
    img_sum = np.sum(array_of_speak, axis=0)
    print('summation of 3 wavelength for speak', img_sum.shape)
    img_plot = np.reshape(img_sum, (128, 384))
    print('After summing up 3 wavelengths, the image shape is', img_plot.shape)
    # plt.imshow(img_plot, cmap="jet")
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
str: str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_2'


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

print('Shape of whole test datasets', np.shape(testdatasets))                                                       # 12, 49152, 441
print('\n')
print('End of printing the test datasets\n')

# # Converting list of test dataset into an array
# testdatasets = np.array(testdatasets)
#
# # reshape the testdataset to feed into the PCA
# reshaped_testdataset = np.reshape(testdatasets, (-1, testdatasets.shape[-1]))
# print("shape of reshaped_dataset:", reshaped_testdataset.shape)                                                     # (12*49152), 441
#
# # Retrieving Principal Components
# pca = PCA(n_components=3)
# pca_testdataset = pca.fit_transform(reshaped_testdataset)
# print("shape of reshaped_dataset after applying pca:", pca_testdataset.shape)                                       # 589824, 3
#
# # Reshape back
# testdatasets = np.reshape(pca_testdataset, (testdatasets.shape[0], testdatasets.shape[1], 3))
# print('Shape of whole test datasets', np.shape(testdatasets))                                                       # 12, 49152, 3
#
# # Transpose the dataset to feed into Resnet 18
# testdatasets = np.transpose(testdatasets, (0, 2, 1))
# print('Shape of whole test datasets after being transposed', np.shape(testdatasets))
#
# # reshape the dataset to feed into the ResNet 18
# testdatasets = np.reshape(testdatasets, (testdatasets.shape[0], testdatasets.shape[1], 128, 384))
# print("shape of final testdataset after being reshaped:", testdatasets.shape)

