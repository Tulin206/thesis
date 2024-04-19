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

# Storing the mean spectrum for soft texture and hard texture separately
class_0 = []
class_1 = []

class_0_max = []
class_1_max = []

max_absorb_list = []
mean_absorb_list = []
wavelength_value_list = []
wavelength_list = []

def load_training_data(array_of_speak,
                       dictionary_of_TrainingSet,
                       transposed_array_of_speak,
                       training_set_num,
                       NumpyArray_of_TrainingSet,
                       file_path, class_label=None):

    array_of_speak = dictionary_of_TrainingSet["spek"]
    x = dictionary_of_TrainingSet["x"]
    print("Shape of spek:", array_of_speak.shape)
    # print(array_of_speak[0])

    # Transpose the input image
    transposed_array_of_speak = np.transpose(array_of_speak)
    print("Shape of spek after being transposed:", transposed_array_of_speak.shape)
    # print(transposed_array_of_speak[0])
    # print(transposed_array_of_speak.shape[0])
    # print(transposed_array_of_speak.shape[1])

    # # Number of spectra in the image
    # num_spectra = transposed_array_of_speak.shape[0]  # transposed_array_of_speak.shape[0] = 49152
    #
    # # Plot all spectra
    # for i in range(num_spectra):
          # Absorbent values those lie on the 441 wavelength (441 absorbent value)
    #     spectrum = transposed_array_of_speak[i]  # transposed_array_of_speak.shape = 49152, 441
    #     wavelengths = range(len(spectrum))  # wavelengths = 441
    #     plt.plot(wavelengths, spectrum)  # Per wavelength (441 wavenummer), 441 absorbent values (1 spectrum) will be plotted.
    #
    # plt.xlabel('Wavelengths')
    # plt.ylabel('Intensity')
    # plt.title('All Spectra')
    # plt.show()
    #
    # # Select a spectrum from the spectroscopic image (e.g., the first spectrum)
    # spectrum = transposed_array_of_speak[0]
    #
    # # Plot the spectrum
    # wavelengths = range(len(spectrum))
    # plt.plot(wavelengths, spectrum)
    # plt.xlabel('Wavelengths')
    # plt.ylabel('Intensity')
    # plt.title('Spectrum')
    # plt.show()


    # For visualize the original image
    img_sum_Original = np.sum(array_of_speak, axis=0)
    print('summation of 441 wavelength for speak', img_sum_Original.shape)
    print('\n')

    reshaped_Original_Image = np.reshape(img_sum_Original, (128, 384))
    print("After reshaping the shape of original image", reshaped_Original_Image.shape)

    # plt.imshow(reshaped_Original_Image, cmap="jet")
    # plt.show()
    print('//////////////////\n')


    # Find the maximum absorbance value for each spectrum
    max_absorbance = np.max(array_of_speak, axis=1)
    print("max_absorbance value:", max_absorbance)
    print("Shape of maximum absorbent value in spek:", max_absorbance.shape)
    wavelength_value = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
    # wavelengths = range(len(max_absorbance))  # wavelengths = 441
    plt.plot(wavelength_value, max_absorbance)
    plt.xlabel('Wavelengths')
    plt.ylabel('Absorbance')
    plt.title('Max absorbent Spectra')
    plt.show()

    # Append max absorbance of all the images into a list
    max_absorb_list.append(max_absorbance)

    # Find the mean absorbance value for each spectrum
    mean_absorbance = np.mean(array_of_speak, axis=1)
    print("min_absorbance value:", mean_absorbance)
    print("Shape of minimum absorbent value in spek:", mean_absorbance.shape)
    plt.plot(wavelength_value, mean_absorbance)
    plt.xlabel('Wavelengths')
    plt.ylabel('Absorbance')
    plt.title('Mean absorbent Spectra')
    plt.show()

    # # Append min absorbance of all the images into a list
    mean_absorb_list.append(mean_absorbance)

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

    # # Plot the new image
    # plt.imshow(new_image, cmap='jet')
    # plt.colorbar(label='Intensity')
    # plt.xlabel('Wavenumber')
    # plt.ylabel('Spectrum Index')
    # plt.title('New Image with Removed Spectra')
    # plt.show()
    print('//////////////////\n')




    # # define standard scaler
    # scaler = StandardScaler()
    #
    # # transform data
    # scaled_original_data = scaler.fit_transform(array_of_speak.T)
    # print("Shape of spek after being transformed and scaled:", scaled_original_data.shape)
    #
    # pca = PCA(n_components=30)
    # original_data_pca = pca.fit_transform(scaled_original_data)
    # print("Shape of spek after having PCA:", original_data_pca.shape)
    #
    # # For visualize the PCA image with 30 components
    # original_img_sum_pca = np.sum(original_data_pca, axis=1)
    # print('summation of 30 PCs for speak', original_img_sum_pca.shape)
    # print('\n')
    #
    # # Plotting image with 30 PC
    # original_img_plot_pca = np.reshape(original_img_sum_pca, (128, 384))
    # print('After summing up 30 PCs and reshaping, the image shape is', original_img_plot_pca.shape)
    # plt.imshow(original_img_plot_pca, cmap="jet")
    # plt.show()
    # print('//////////////////\n')



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
    img_2d = np.reshape(reshaped_Original_Image, (-1, 1))  # reshaped_Original_Image
    print(img_2d.shape)

    # Apply K-means clustering
    kmeans = KMeans(n_init = 10, n_clusters=2, random_state=0)
    kmeans.fit(img_2d)

    # Get the labels assigned to each pixel
    labels = kmeans.labels_

    # Reshape the labels back to the image shape
    cluster_assignments = np.reshape(labels, (128, 384))

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    print(cluster_centers.shape)

    # # Visualize the clustered image
    # plt.imshow(cluster_assignments, cmap="jet")
    # plt.title("Clustered Image")
    # plt.colorbar()
    # plt.show()



    # Apply K-means clustering
    kmeans0 = KMeans(n_init=10, n_clusters=2, random_state=0)
    kmeans0.fit(reshaped_Original_Image)

    # Get the labels assigned to each pixel
    labels0 = kmeans0.labels_

    # # plot the centroid of each cluster
    # plt.scatter(kmeans0.cluster_centers_[:, 0], kmeans0.cluster_centers_[:, 1], color='green', marker='*',
    #             label='centroid', s=200)
    #
    # # Separate the data points based on cluster labels
    # cluster_0 = reshaped_Original_Image[labels0 == 0]
    # cluster_1 = reshaped_Original_Image[labels0 == 1]
    #
    # # Create a scatter plot for each cluster
    # plt.scatter(cluster_0[:, 0], cluster_0[:, 1], c='red', label='Cluster 0')
    # plt.scatter(cluster_1[:, 0], cluster_1[:, 1], c='black', label='Cluster 1')
    # # Set labels and title
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.title('K-means Clustering')
    #
    # # Add a legend
    # plt.legend()
    #
    # # Show the plot
    # plt.show()



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


    # define standard scaler
    scaler = StandardScaler()

    # transform data
    scaled_new_image = scaler.fit_transform(new_image)
    print("Shape of new_image that is stored to feed into the deep model:", scaled_new_image.shape)


    # Covert dataset into array
    NumpyArray_of_TrainingSet = np.array(scaled_new_image)



    def save_array_to_file(array_to_save: np.ndarray, file_path: str):
        np.save(file_path, array_to_save)

    save_array_to_file(NumpyArray_of_TrainingSet, file_path)


    if class_label == 0:
        class_0.append(mean_absorbance)
    else:
        class_1.append(mean_absorbance)

    if class_label == 0:
        class_0_max.append(max_absorbance)
    else:
        class_1_max.append(max_absorbance)


# Transfer data from matlab file into Python
training20 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_20_neu.mat")

print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)
wavelength_list.append(wavelength_values)
wavelength_value_list.append(wavelength_values)

# Variable Declaration
t20 = []
t20_tr = []
training_set_1 = []
np_tarray_1 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_1'


# function calling
load_training_data(t20, training20, t20_tr, training_set_1, np_tarray_1, str, 0)

# Transfer data from matlab file into Python
training22 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_22_neu.mat")

print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# variable Declaration
t22 = []
t22_tr = []
training_set_2 = []
np_tarray_2 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_2'

# function calling
load_training_data(t22, training22, t22_tr, training_set_2, np_tarray_2, str, 0)

# Transfer data from matlab file into Python
training23 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_23_neu.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t23 = []
t23_tr = []
training_set_3 = []
np_tarray_3 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_3'

# function calling
load_training_data(t23, training23, t23_tr, training_set_3, np_tarray_3, str, 1)


# Transfer data from matlab file into Python
training25 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_25_neu.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t25 = []
t25_tr = []
training_set_4 = []
np_tarray_4 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_4'

# function calling
load_training_data(t25, training25, t25_tr, training_set_4, np_tarray_4, str, 1)


# Transfer data from matlab file into Python
training28 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_28_neu2.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t28 = []
t28_tr = []
training_set_5 = []
np_tarray_5 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_5'

# function calling
load_training_data(t28, training28, t28_tr, training_set_5, np_tarray_5, str, 1)


# Transfer data from matlab file into Python
training29 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_29_neu2.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t29 = []
t29_tr = []
training_set_6 = []
np_tarray_6 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_6'

# function calling
load_training_data(t29, training29, t29_tr, training_set_6, np_tarray_6, str, 0)


# Transfer data from matlab file into Python
training31 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_31_neu.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t31 = []
t31_tr = []
training_set_7 = []
np_tarray_7 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_7'

# function calling
load_training_data(t31, training31, t31_tr, training_set_7, np_tarray_7, str, 1)


# Transfer data from matlab file into Python
training32 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_32_neu.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t32 = []
t32_tr = []
training_set_8 = []
np_tarray_8 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_8'

# function calling
load_training_data(t32, training32, t32_tr, training_set_8, np_tarray_8, str, 0)


# Transfer data from matlab file into Python
training33 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_33_neuII.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t33 = []
t33_tr = []
training_set_9 = []
np_tarray_9 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_9'

# function calling
load_training_data(t33, training33, t33_tr, training_set_9, np_tarray_9, str, 1)


# Transfer data from matlab file into Python
training34 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_34_neuII.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t34 = []
t34_tr = []
training_set_10 = []
np_tarray_10 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_10'

# function calling
load_training_data(t34, training34, t34_tr, training_set_10, np_tarray_10, str, 1)

# Transfer data from matlab file into Python
training1 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_I_Sample_1_neu_korr.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t1 = []
t1_tr = []
training_set_11 = []
np_tarray_11 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_11'

# function calling
load_training_data(t1, training1, t1_tr, training_set_11, np_tarray_1, str, 0)

# Transfer data from matlab file into Python
training3 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_I_Sample_3_neu2.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t3 = []
t3_tr = []
training_set_12 = []
np_tarray_12 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_12'

# function calling
load_training_data(t3, training3, t3_tr, training_set_12, np_tarray_12, str, 1)

# Transfer data from matlab file into Python
training7 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_7_neu.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t7 = []
t7_tr = []
training_set_13 = []
np_tarray_13 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_13'

# function calling
load_training_data(t7, training7, t7_tr, training_set_13, np_tarray_13, str, 1)

# Transfer data from matlab file into Python
training10 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_III_Sample_10_korr.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t10 = []
t10_tr = []
training_set_14 = []
np_tarray_14 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_14'

# function calling
load_training_data(t10, training10, t10_tr, training_set_14, np_tarray_14, str, 1)

# Transfer data from matlab file into Python
training11 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_III_Sample_11.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t11 = []
t11_tr = []
training_set_15 = []
np_tarray_15 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_15'

# function calling
load_training_data(t11, training11, t11_tr, training_set_15, np_tarray_15, str, 1)

# Transfer data from matlab file into Python
training17 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_V_Sample_17_korr.mat")
print("type:", type(training20))
print("len:", len(training20))
print("keys:", training20.keys())

x = training20["x"]
print("Shape of x:", x.shape)

# Assuming x is an array with shape (1, 441)
wavelength_values = x[0, :]  # Extract the first row (row 0) which contains the wavelength values
print("shape of wavelength:", wavelength_values.shape)
print("441 wavelength:", wavelength_values)

wavelength_value_list.append(wavelength_values)

# Variable Declaration
t17 = []
t17_tr = []
training_set_16 = []
np_tarray_16 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_16'

# function calling
load_training_data(t17, training17, t17_tr, training_set_16, np_tarray_16, str, 0)

# Load the 10 training spectral datasets into a list
datasets = [np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_1.npy'), # soft texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_2.npy'), # soft texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_3.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_4.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_5.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_6.npy'), # soft texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_7.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_8.npy'), # soft texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_9.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_10.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_11.npy'), # soft texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_12.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_13.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_14.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_15.npy'), # hard texture
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PycharmProjects_dataset.npy/trainingset_16.npy')] # soft texture

print('Shape of whole training datasets', np.shape(datasets))
print('\n')
print('End of printing the training datasets\n')

print("shape of mean_absorb_list:", np.shape(mean_absorb_list))

# Create a stacked bar plot
plt.figure(figsize=(12, 6))
bottom = np.zeros(441)  # Initialize the bottom of the bars as zeros

for i in range(16):
    plt.bar(wavelength_values, mean_absorb_list[i], bottom=bottom, label=f'Sample {i+1}')
    bottom += mean_absorb_list[i]  # Update the bottom for the next set of bars

plt.xlabel('Wavelength')
plt.ylabel('Mean Absorbance Value')
plt.title('Stacked Bar Plot of Mean Absorbance Values vs. Wavelength')
plt.legend()
plt.grid(True)
plt.show()

# Create a box plot for the mean absorbance values
plt.figure(figsize=(8, 6))
plt.boxplot(mean_absorb_list)
plt.title("Box Plot of Mean Absorbance Values")
plt.xlabel("Image Index")
plt.ylabel("Mean Absorbance")
plt.show()

# Create a box plot for the max absorbance values
plt.figure(figsize=(8, 6))
plt.boxplot(max_absorb_list)
plt.title("Box Plot of Max Absorbance Values")
plt.xlabel("Image Index")
plt.ylabel("Max Absorbance")
plt.show()

# Convert the list of mean absorbance arrays into a 2D NumPy array
mean_absorb_array = np.vstack(mean_absorb_list)

# Plot the box plots
plt.figure(figsize=(12, 8))
plt.boxplot(mean_absorb_array.T, vert=False)
plt.xticks(range(1, len(wavelength_values[:10]) + 1), wavelength_values[:10])
plt.title("Box Plot of Mean Absorbance Values for Each Wavelength")
plt.xlabel("wavelength")
plt.ylabel("Mean Absorbance")
plt.show()


# Convert the lists to NumPy arrays for easier manipulation
class_0 = np.array(class_0)
class_1 = np.array(class_1)
class_0_max = np.array(class_0_max)
class_1_max = np.array(class_1_max)
wavelength_list = np.array(wavelength_list).flatten()  # Flatten the wavelength list as it has shape (1, 441)

# Create a figure and axis for the plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot the spectra for class_0 (blue)
for i in range(min(10, len(class_0))):  # Assuming 10 samples in class_0
    plt.plot(wavelength_list, class_0[i], color='blue', alpha=0.5)  # Alpha for transparency

# Plot the spectra for class_1 (red)
for i in range(min(6, len(class_1))):  # Assuming 6 samples in class_1
    plt.plot(wavelength_list, class_1[i], color='red', alpha=0.5)  # Alpha for transparency

# Customize the plot
plt.xlabel('Wavelength')
plt.ylabel('Mean Absorbance')
plt.title('Mean Absorbance Spectra for Class 0 and Class 1')
plt.grid(True)
# plt.legend(['Class 0 (Blue)', 'Class 1 (Red)'])  # Add a legend


# Create custom legend handles and labels
legend_handles = [
    plt.Line2D([0], [0], color='blue', lw=2),
    plt.Line2D([0], [0], color='red', lw=2)
]
legend_labels = ['Class 0 (Soft_Texture)', 'Class 1 (Hard_Texture)']

# Add a legend with custom handles and labels
plt.legend(legend_handles, legend_labels)

# Show the plot
plt.show()


# Create a figure and axis for the plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot the spectra for class_0 (blue)
for i in range(min(10, len(class_0_max))):  # Assuming 10 samples in class_0
    plt.plot(wavelength_list, class_0_max[i], color='blue', alpha=0.5)  # Alpha for transparency

# Plot the spectra for class_1 (red)
for i in range(min(6, len(class_1_max))):  # Assuming 6 samples in class_1
    plt.plot(wavelength_list, class_1_max[i], color='red', alpha=0.5)  # Alpha for transparency

# Customize the plot
plt.xlabel('Wavelength')
plt.ylabel('Mean Absorbance')
plt.title('Max Absorbance Spectra for Class 0 and Class 1')
plt.grid(True)
# plt.legend(['Class 0 (Blue)', 'Class 1 (Red)'])  # Add a legend


# Create custom legend handles and labels
legend_handles = [
    plt.Line2D([0], [0], color='blue', lw=2),
    plt.Line2D([0], [0], color='red', lw=2)
]
legend_labels = ['Class 0 (Soft_Texture)', 'Class 1 (Hard_Texture)']

# Add a legend with custom handles and labels
plt.legend(legend_handles, legend_labels)

# Show the plot
plt.show()