import os
import scipy
from scipy.io import loadmat
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.decomposition import PCA

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import pandas as pd

import lmfit
from lmfit.models import LinearModel

from scipy.signal import savgol_filter

def load_test_data(array_of_speak,
                       dictionary_of_TrainingSet,
                       transposed_array_of_speak,
                       training_set_num,
                       NumpyArray_of_TrainingSet,
                       file_path):

    array_of_speak = dictionary_of_TrainingSet["spek"]
    print("Shape of spek:", array_of_speak.shape)
    #print(array_of_speak[0])


    # For visualize the original image
    img_sum_Original = np.sum(array_of_speak, axis=0)
    print('summation of 441 wavelength for speak', img_sum_Original.shape)
    print('\n')

    reshaped_Original_Image = np.reshape(img_sum_Original, (128, 384))
    print("After reshaping the shape of original image", reshaped_Original_Image.shape)

    plt.imshow(reshaped_Original_Image, cmap="jet")
    plt.show()
    print('//////////////////\n')





    # Find the maximum absorbance value for each spectrum
    max_absorbance = np.max(array_of_speak, axis=0)
    print("Shape of maximum absorbent value in spek:", max_absorbance.shape)

    # Find the indices of spectra that meet the condition
    indices_to_remove = np.where((max_absorbance > 2) | (max_absorbance < 0.5))

    # Remove the spectra from the dataset
    filtered_spectra_data = np.delete(array_of_speak, indices_to_remove, axis=1)

    # Print the filtered data shape
    print("Filtered data shape:", filtered_spectra_data.shape)

    # Create a new image with the original shape and set all spectra to the original values
    new_image = np.copy(array_of_speak)
    print("Shape of new image that is copied from original image", new_image.shape)

    # Transpose the input image
    transposed_new_image = np.transpose(new_image)
    print("Shape of new image after being transposed:", transposed_new_image.shape)
    # print(transposed_array_of_speak[0])

    # For visualize the original image
    img_sum_CopiedImage = np.sum(transposed_new_image, axis=1)
    print('summation of 441 wavelength for copied image', img_sum_CopiedImage.shape)
    print('\n')

    reshaped_new_image = np.reshape(img_sum_CopiedImage, (128, 384))
    print("After reshaping the shape of copied image", reshaped_new_image.shape)

    # Set the removed spectra to 0 (white) in the new image
    reshaped_new_image.ravel()[indices_to_remove] = np.nan

    # Reshape the new image back to the original shape
    new_image = np.reshape(reshaped_new_image, reshaped_Original_Image.shape)
    print("Shape of copied image after filtering out the outliers and making them white:", new_image.shape)

    # Plot the new image
    plt.imshow(new_image, cmap='jet')
    plt.colorbar(label='Intensity')
    plt.xlabel('Wavenumber')
    plt.ylabel('Spectrum Index')
    plt.title('New Image with Removed Spectra')
    plt.show()
    print('//////////////////\n')




    # define standard scaler
    scaler = StandardScaler()

    # transform data
    scaled_original_data = scaler.fit_transform(array_of_speak.T)
    print("Shape of spek after being transformed and scaled:", scaled_original_data.shape)

    pca = PCA(n_components=30)
    original_data_pca = pca.fit_transform(scaled_original_data)
    print("Shape of spek after having PCA:", original_data_pca.shape)

    # For visualize the PCA image with 30 components
    original_img_sum_pca = np.sum(original_data_pca, axis=1)
    print('summation of 30 PCs for speak', original_img_sum_pca.shape)
    print('\n')

    # Plotting image with 30 PC
    original_img_plot_pca = np.reshape(original_img_sum_pca, (128, 384))
    print('After summing up 30 PCs and reshaping, the image shape is', original_img_plot_pca.shape)
    plt.imshow(original_img_plot_pca, cmap="jet")
    plt.show()
    print('//////////////////\n')



    # # transform data
    # scaled_filtered_data = scaler.fit_transform(filtered_spectra_data.T)
    # print("Shape of filtered image after being scaled:", scaled_filtered_data.shape)
    #
    # pca = PCA(n_components=30)
    # data_pca = pca.fit_transform(scaled_filtered_data)
    # print("Shape of filtered image after having PCA:", data_pca.shape)

    # # Get the first principal component
    # first_pc = data_pca[:, 1]
    # print(first_pc.shape)

    # # For visualize the PCA image with first PC
    # img_sum_pca_1 = np.sum(first_pc, axis=1)
    # print('summation of 1st PC for speak', img_sum_pca_1.shape)
    # print('\n')

    # # Plotting image with 1st PC
    # img_plot_pca_1 = np.reshape(first_pc, (128, 384))
    # print('After summing up 1st PC and reshaping, the image shape is', img_plot_pca_1.shape)
    # plt.imshow(img_plot_pca_1, cmap="jet")
    # plt.show()
    # print('//////////////////\n')


    # # For visualize the PCA image with 30 components
    # img_sum_pca = np.sum(data_pca, axis=1)
    # print('summation of 30 PCs for speak', img_sum_pca.shape)
    # print('\n')
    #
    #
    # # Plotting image with 30 PC
    # img_plot_pca = np.reshape(img_sum_pca, (43269, 1))
    # print('After summing up 30 PCs and reshaping, the image shape is', img_plot_pca.shape)
    # plt.imshow(img_plot_pca, cmap="jet", aspect='auto')
    # plt.show()
    # print('//////////////////\n')

    # # For visualize the original image
    # img_sum = np.sum(array_of_speak, axis=0)
    # print('summation of 441 wavelength for speak', img_sum.shape)
    # print('\n')

    # # Apply Savitzky-Golay smoothing filter to smooth the spectrum
    # smoothed_spectrum = savgol_filter(img_sum, window_length=101, polyorder=3)
    #
    # # Subtract the smoothed spectrum from the original summed spectrum to obtain the baseline-corrected spectrum
    # baseline_corrected = img_sum - smoothed_spectrum
    #
    # img_plot = np.reshape(baseline_corrected, (128, 384))
    # print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
    # plt.imshow(img_plot, cmap="jet")
    # plt.show()
    # print('//////////////////\n')


    # Reshape the image to a 2D array
    img_2d = np.reshape(original_img_plot_pca, (-1, 1))  # reshaped_Original_Image
    print(img_2d.shape)

    # Apply K-means clustering
    kmeans = KMeans(n_init = 10, n_clusters=3, random_state=0)
    kmeans.fit(img_2d)

    # Get the labels assigned to each pixel
    labels = kmeans.labels_

    # Reshape the labels back to the image shape
    cluster_assignments = np.reshape(labels, (128, 384))

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Visualize the clustered image
    plt.imshow(cluster_assignments, cmap="jet")
    plt.title("Clustered Image")
    plt.colorbar()
    plt.show()

    # # Create a copy of the new_image
    # clustered_image = np.copy(new_image)
    #
    # # Reshape the image to a 2D array
    # img_2d = np.reshape(clustered_image, (-1, 1))
    #
    # # Find the indices of non-NaN (valid) pixels
    # valid_indices = ~np.isnan(img_2d)
    #
    # # Apply k-means clustering on valid pixels
    # kmeans = KMeans(n_clusters=3)
    # kmeans.fit(img_2d[valid_indices])
    #
    # # Get the labels assigned to each pixel
    # labels = kmeans.predict(img_2d[valid_indices].reshape(-1, 1))
    #
    # # Assign cluster labels to valid pixels in the clustered_image
    # clustered_image[valid_indices] = labels
    #
    # # Plot the clustered image
    # plt.imshow(clustered_image, cmap='jet')
    # plt.colorbar(label='Cluster Label')
    # plt.xlabel('Wavenumber')
    # plt.ylabel('Spectrum Index')
    # plt.title('K-means Clustering of New Image')
    # plt.show()



    # transform data
    scaled_new_image = scaler.fit_transform(new_image)
    print("Shape of new_image that is stored to feed into the deep model:", scaled_new_image.shape)


    # Covert dataset into array
    NumpyArray_of_TrainingSet = np.array(scaled_new_image)



    def save_array_to_file(array_to_save: np.ndarray, file_path: str):
        np.save(file_path, array_to_save)

    save_array_to_file(NumpyArray_of_TrainingSet, file_path)



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

print('Shape of whole test datasets', np.shape(testdatasets))
print('\n')
print('End of printing the test datasets\n')
