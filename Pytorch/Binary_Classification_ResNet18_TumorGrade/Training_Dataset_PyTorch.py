import os
import scipy
from scipy.io import loadmat
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Set to 1 for blocking CUDA calls (useful for debugging)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set to the GPU device index you want to use

import numpy as np

import torch
torch.cuda.memory_reserved(0)  # Set the GPU index as needed

import random

# Set seed.
seed = 180
seed = 90
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# Function to convert matlab into python
# def load_matlab(dictionary_of_trainingSet):
#     print("Convert matlab data into python dataset")
#     dictionary_of_TrainingSet = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_20_neu.mat")
#
# training20 = {}
# load_matlab(training20)
#
# print("type:", type(training20))
# print("len:", len(training20))
# print("keys:", training20.keys())

# Define function to load the training data
def load_training_data(array_of_speak,
                       dictionary_of_TrainingSet,
                       transposed_array_of_speak,
                       training_set_num,
                       NumpyArray_of_TrainingSet,
                       file_path):

    array_of_speak = dictionary_of_TrainingSet["spek"]
    print("Shape of spek:", array_of_speak.shape)
    # print(array_of_speak[0])
    # print(array_of_speak[1])

    # Transpose the input image
    transposed_array_of_speak = np.transpose(array_of_speak)
    print("Shape of spek after being transposed:", transposed_array_of_speak.shape)
    # print(transposed_array_of_speak[0])
    # print(transposed_array_of_speak.shape[0])
    # print(transposed_array_of_speak.shape[1])


    # Reshape input image
    training_set_num = np.reshape(transposed_array_of_speak, (128, 384, 441))
    # training_set_num = np.reshape(array_of_speak, (441, 128, 384))
    # training_set_num = array_of_speak.reshape(441, 128, 384)
    # # dataset_1 = np.reshape(img_plot, (4, 12, 1024))
    # NumpyArray_of_TrainingSet = np.array(training_set_num)
    print('After reshaping the image dimension', training_set_num.shape)
    training_set_num = np.array(training_set_num)

    # save the array as an .npy file
    # training_set_num = np.save('NumpyFile_of_TrainingSet.npy', NumpyArray_of_TrainingSet)

    def save_array_to_file(array_to_save: np.ndarray, file_path: str):
        np.save(file_path, array_to_save)

    save_array_to_file(training_set_num, file_path)

    # # For visualize the original image
    # img_sum = np.sum(array_of_speak, axis=0)
    # print('summation of 441 wavelength for speak', img_sum.shape)
    # img_plot = np.reshape(img_sum, (1, 128, 384))
    # save_array_to_file(img_plot, file_path)
    # print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
    # img_plot = np.reshape(img_plot, (1, 128*384))
    # img_plot = np.sum(img_plot, axis=0)
    # img_plot = np.reshape(img_plot, (128, 384))
    # plt.imshow(img_plot, cmap="jet")
    # plt.show()
    print('//////////////////\n')


# Transfer data from matlab file into Python
training20 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_20_neu.mat")
# training20 = loadmat(r"/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Test files/Training_Data/Sample_20_neu.mat")

# Variable Declaration
t20 = []
t20_tr = []
training_set_1 = []
np_tarray_1 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_1'


# function calling
load_training_data(t20, training20, t20_tr, training_set_1, np_tarray_1, str)

# Transfer data from matlab file into Python
training22 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_22_neu.mat")

# variable Declaration
t22 = []
t22_tr = []
training_set_2 = []
np_tarray_2 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_2'

# function calling
load_training_data(t22, training22, t22_tr, training_set_2, np_tarray_2, str)

# Transfer data from matlab file into Python
training23 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_23_neu.mat")

# Variable Declaration
t23 = []
t23_tr = []
training_set_3 = []
np_tarray_3 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_3'

# function calling
load_training_data(t23, training23, t23_tr, training_set_3, np_tarray_3, str)


# Transfer data from matlab file into Python
training25 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_25_neu.mat")

# Variable Declaration
t25 = []
t25_tr = []
training_set_4 = []
np_tarray_4 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_4'

# function calling
load_training_data(t25, training25, t25_tr, training_set_4, np_tarray_4, str)


# Transfer data from matlab file into Python
training28 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_28_neu2.mat")

# Variable Declaration
t28 = []
t28_tr = []
training_set_5 = []
np_tarray_5 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_5'

# function calling
load_training_data(t28, training28, t28_tr, training_set_5, np_tarray_5, str)


# Transfer data from matlab file into Python
training29 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_29_neu2.mat")

# Variable Declaration
t29 = []
t29_tr = []
training_set_6 = []
np_tarray_6 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_6'

# function calling
load_training_data(t29, training29, t29_tr, training_set_6, np_tarray_6, str)


# Transfer data from matlab file into Python
training31 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_31_neu.mat")

# Variable Declaration
t31 = []
t31_tr = []
training_set_7 = []
np_tarray_7 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_7'

# function calling
load_training_data(t31, training31, t31_tr, training_set_7, np_tarray_7, str)


# Transfer data from matlab file into Python
training32 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_32_neu.mat")

# Variable Declaration
t32 = []
t32_tr = []
training_set_8 = []
np_tarray_8 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_8'

# function calling
load_training_data(t32, training32, t32_tr, training_set_8, np_tarray_8, str)


# Transfer data from matlab file into Python
training33 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_33_neuII.mat")

# Variable Declaration
t33 = []
t33_tr = []
training_set_9 = []
np_tarray_9 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_9'

# function calling
load_training_data(t33, training33, t33_tr, training_set_9, np_tarray_9, str)


# Transfer data from matlab file into Python
training34 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_34_neuII.mat")

# Variable Declaration
t34 = []
t34_tr = []
training_set_10 = []
np_tarray_10 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_10'

# function calling
load_training_data(t34, training34, t34_tr, training_set_10, np_tarray_10, str)

# Transfer data from matlab file into Python
training1 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_I_Sample_1_neu_korr.mat")

# Variable Declaration
t1 = []
t1_tr = []
training_set_11 = []
np_tarray_11 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_11'

# function calling
load_training_data(t1, training1, t1_tr, training_set_11, np_tarray_1, str)

# Transfer data from matlab file into Python
training3 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_I_Sample_3_neu2.mat")

# Variable Declaration
t3 = []
t3_tr = []
training_set_12 = []
np_tarray_12 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_12'

# function calling
load_training_data(t3, training3, t3_tr, training_set_12, np_tarray_12, str)

# Transfer data from matlab file into Python
training7 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_7_neu.mat")

# Variable Declaration
t7 = []
t7_tr = []
training_set_13 = []
np_tarray_13 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_13'

# function calling
load_training_data(t7, training7, t7_tr, training_set_13, np_tarray_13, str)

# Transfer data from matlab file into Python
training10 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_III_Sample_10_korr.mat")

# Variable Declaration
t10 = []
t10_tr = []
training_set_14 = []
np_tarray_14 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_14'

# function calling
load_training_data(t10, training10, t10_tr, training_set_14, np_tarray_14, str)

# Transfer data from matlab file into Python
training11 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_III_Sample_11.mat")

# Variable Declaration
t11 = []
t11_tr = []
training_set_15 = []
np_tarray_15 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_15'

# function calling
load_training_data(t11, training11, t11_tr, training_set_15, np_tarray_15, str)

# Transfer data from matlab file into Python
training17 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_V_Sample_17_korr.mat")

# Variable Declaration
t17 = []
t17_tr = []
training_set_16 = []
np_tarray_16 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_16'

# function calling
load_training_data(t17, training17, t17_tr, training_set_16, np_tarray_16, str)

# Transfer data from matlab file into Python
test_data_I2 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_I_Sample_2_korr.mat")

# Variable Declaration
test_I2 = []
test_I2_tr = []
test_dataset_2 = []
np_darray_2  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_1'

#### 12 test samples


# function calling
load_training_data(test_I2, test_data_I2, test_I2_tr, test_dataset_2, np_darray_2 , str)


# Transfer data from matlab file into Python
test_data_II8 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_II_Sample_8_neu_korr.mat")

# Variable Declaration
test_II8 = []
test_II8_tr = []
test_dataset_3 = []
np_darray_3  = []
str: str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_2'


# function calling
load_training_data(test_II8, test_data_II8, test_II8_tr, test_dataset_3, np_darray_3 , str)


# Transfer data from matlab file into Python
test_data_IV13 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_IV_Sample_13.mat")


# Variable Declaration
test_IV13 = []
test_IV13_tr = []
test_dataset_8 = []
np_darray_8  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_3'


# function calling
load_training_data(test_IV13, test_data_IV13, test_IV13_tr, test_dataset_8, np_darray_8, str)


# Transfer data from matlab file into Python
test_data_IV14 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Slide_IV_Sample_14.mat")

# Variable Declaration
test_IV14 = []
test_IV14_tr = []
test_dataset_9 = []
np_darray_9  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_4'


# function calling
load_training_data(test_IV14, test_data_IV14, test_IV14_tr, test_dataset_9, np_darray_9, str)


# Transfer data from matlab file into Python
test_data_4 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_4_neuII.mat")

# Variable Declaration
test_4 = []
test_4_tr = []
test_dataset_10 = []
np_darray_10  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_5'


# function calling
load_training_data(test_4, test_data_4, test_4_tr, test_dataset_10, np_darray_10, str)


# Transfer data from matlab file into Python
test_data_6 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_6_neuII.mat")

# Variable Declaration
test_6 = []
test_6_tr = []
test_dataset_11 = []
np_darray_11  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_6'


# function calling
load_training_data(test_6, test_data_6, test_6_tr, test_dataset_11, np_darray_11, str)


# Transfer data from matlab file into Python
test_data_9 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_9_neuII.mat")

# Variable Declaration
test_9 = []
test_9_tr = []
test_dataset_13 = []
np_darray_13  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_7'


# function calling
load_training_data(test_9, test_data_9, test_9_tr, test_dataset_13, np_darray_13, str)


# Transfer data from matlab file into Python
test_data_15 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_15_neu.mat")

# Variable Declaration
test_15 = []
test_15_tr = []
test_dataset_14 = []
np_darray_14  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_8'


# function calling
load_training_data(test_15, test_data_15, test_15_tr, test_dataset_14, np_darray_14, str)


# Transfer data from matlab file into Python
test_data_16 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_16_neu.mat")

# Variable Declaration
test_16 = []
test_16_tr = []
test_dataset_15 = []
np_darray_15  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_9'


# function calling
load_training_data(test_16, test_data_16, test_16_tr, test_dataset_15, np_darray_15, str)



# Transfer data from matlab file into Python
test_data_18 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_18_neu.mat")

# Variable Declaration
test_18 = []
test_18_tr = []
test_dataset_16 = []
np_darray_16  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_10'


# function calling
load_training_data(test_18, test_data_18, test_18_tr, test_dataset_16, np_darray_16, str)


# Transfer data from matlab file into Python
test_data_19 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_19_neu.mat")

# Variable Declaration
test_19 = []
test_19_tr = []
test_dataset_17 = []
np_darray_17  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_11'


# function calling
load_training_data(test_19, test_data_19, test_19_tr, test_dataset_17, np_darray_17, str)


# Transfer data from matlab file into Python
test_data_21 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Test_Data/Sample_21_neu.mat")

# Variable Declaration
test_21 = []
test_21_tr = []
test_dataset_18 = []
np_darray_18  = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_12'


# function calling
load_training_data(test_21, test_data_21, test_21_tr, test_dataset_18, np_darray_18, str)

# Load the 10 training spectral datasets into a list
datasets = [np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_1.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_2.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_3.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_4.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_5.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_6.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_7.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_8.npy'),
            # np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_9.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_10.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_11.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_12.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_13.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_14.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_15.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_16.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/testdataset_1.npy'),
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

datasets = np.transpose(datasets, (0, 3, 1, 2))
print('Shape of whole training datasets', np.shape(datasets))
print('\n')
print('End of printing the training datasets\n')