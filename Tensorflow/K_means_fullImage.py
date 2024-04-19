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

def load_training_data(array_of_speak,
                       dictionary_of_TrainingSet,
                       transposed_array_of_speak,
                       training_set_num,
                       NumpyArray_of_TrainingSet,
                       file_path):

    array_of_speak = dictionary_of_TrainingSet["spek"]
    print("Shape of spek:", array_of_speak.shape)

    # # Transpose the input image
    # transposed_array_of_speak = np.transpose(array_of_speak)
    # print("Shape of spek after being transposed:", transposed_array_of_speak.shape)

    # define standard scaler
    scaler = StandardScaler()

    # # transform data with scaling
    # array_of_speak = scaler.fit_transform(array_of_speak)
    # print("Shape of spek after being scaled:", array_of_speak.shape)

    # For visualize the original image
    img_sum = np.sum(array_of_speak, axis=0)
    print('summation of 441 wavelength for speak', img_sum.shape)
    print('\n')



    # img_plot = np.reshape(img_sum, (128, 384))
    # print('After summing up 441 wavelengths, the image shape is', img_plot.shape)
    # plt.imshow(img_plot, cmap="jet")
    # print('//////////////////\n')


    # Covert dataset into array
    NumpyArray_of_TrainingSet = np.array(img_sum)



    def save_array_to_file(array_to_save: np.ndarray, file_path: str):
        np.save(file_path, array_to_save)

    save_array_to_file(NumpyArray_of_TrainingSet, file_path)



# Transfer data from matlab file into Python
training20 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_20_neu.mat")

# Variable Declaration
t20 = []
t20_tr = []
training_set_1 = []
np_tarray_1 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_1'


# function calling
load_training_data(t20, training20, t20_tr, training_set_1, np_tarray_1, str)

# Transfer data from matlab file into Python
training22 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_22_neu.mat")

# variable Declaration
t22 = []
t22_tr = []
training_set_2 = []
np_tarray_2 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_2'

# function calling
load_training_data(t22, training22, t22_tr, training_set_2, np_tarray_2, str)

# Transfer data from matlab file into Python
training23 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_23_neu.mat")

# Variable Declaration
t23 = []
t23_tr = []
training_set_3 = []
np_tarray_3 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_3'

# function calling
load_training_data(t23, training23, t23_tr, training_set_3, np_tarray_3, str)


# Transfer data from matlab file into Python
training25 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_25_neu.mat")

# Variable Declaration
t25 = []
t25_tr = []
training_set_4 = []
np_tarray_4 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_4'

# function calling
load_training_data(t25, training25, t25_tr, training_set_4, np_tarray_4, str)


# Transfer data from matlab file into Python
training28 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_28_neu2.mat")

# Variable Declaration
t28 = []
t28_tr = []
training_set_5 = []
np_tarray_5 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_5'

# function calling
load_training_data(t28, training28, t28_tr, training_set_5, np_tarray_5, str)


# Transfer data from matlab file into Python
training29 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_29_neu2.mat")

# Variable Declaration
t29 = []
t29_tr = []
training_set_6 = []
np_tarray_6 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_6'

# function calling
load_training_data(t29, training29, t29_tr, training_set_6, np_tarray_6, str)


# Transfer data from matlab file into Python
training31 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_31_neu.mat")

# Variable Declaration
t31 = []
t31_tr = []
training_set_7 = []
np_tarray_7 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_7'

# function calling
load_training_data(t31, training31, t31_tr, training_set_7, np_tarray_7, str)


# Transfer data from matlab file into Python
training32 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_32_neu.mat")

# Variable Declaration
t32 = []
t32_tr = []
training_set_8 = []
np_tarray_8 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_8'

# function calling
load_training_data(t32, training32, t32_tr, training_set_8, np_tarray_8, str)


# Transfer data from matlab file into Python
training33 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_33_neuII.mat")

# Variable Declaration
t33 = []
t33_tr = []
training_set_9 = []
np_tarray_9 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_9'

# function calling
load_training_data(t33, training33, t33_tr, training_set_9, np_tarray_9, str)


# Transfer data from matlab file into Python
training34 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_34_neuII.mat")

# Variable Declaration
t34 = []
t34_tr = []
training_set_10 = []
np_tarray_10 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_10'

# function calling
load_training_data(t34, training34, t34_tr, training_set_10, np_tarray_10, str)

# Transfer data from matlab file into Python
training1 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_I_Sample_1_neu_korr.mat")

# Variable Declaration
t1 = []
t1_tr = []
training_set_11 = []
np_tarray_11 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_11'

# function calling
load_training_data(t1, training1, t1_tr, training_set_11, np_tarray_1, str)

# Transfer data from matlab file into Python
training3 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_I_Sample_3_neu2.mat")

# Variable Declaration
t3 = []
t3_tr = []
training_set_12 = []
np_tarray_12 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_12'

# function calling
load_training_data(t3, training3, t3_tr, training_set_12, np_tarray_12, str)

# Transfer data from matlab file into Python
training7 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Sample_7_neu.mat")

# Variable Declaration
t7 = []
t7_tr = []
training_set_13 = []
np_tarray_13 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_13'

# function calling
load_training_data(t7, training7, t7_tr, training_set_13, np_tarray_13, str)

# Transfer data from matlab file into Python
training10 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_III_Sample_10_korr.mat")

# Variable Declaration
t10 = []
t10_tr = []
training_set_14 = []
np_tarray_14 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_14'

# function calling
load_training_data(t10, training10, t10_tr, training_set_14, np_tarray_14, str)

# Transfer data from matlab file into Python
training11 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_III_Sample_11.mat")

# Variable Declaration
t11 = []
t11_tr = []
training_set_15 = []
np_tarray_15 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_15'

# function calling
load_training_data(t11, training11, t11_tr, training_set_15, np_tarray_15, str)

# Transfer data from matlab file into Python
training17 = loadmat(r"/mnt/ceph/tco/TCO-Students/Projects/Spectograpy/Test files/Training_Data/Slide_V_Sample_17_korr.mat")

# Variable Declaration
t17 = []
t17_tr = []
training_set_16 = []
np_tarray_16 = []
str = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_16'

# function calling
load_training_data(t17, training17, t17_tr, training_set_16, np_tarray_16, str)

# Load the 10 training spectral datasets into a list
datasets = [np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_1.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_2.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_3.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_4.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_5.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_6.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_7.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_8.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_9.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_10.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_11.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_12.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_13.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_14.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_15.npy'),
            np.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/k_means/trainingset_16.npy')]
print('\n\n')
print('Shape of whole training datasets', np.shape(datasets))
print('End of printing the training datasets\n')


x = np.array(datasets)
# print(x)


# Retrieve file names using list comprehension
file_names = []
for i, dataset in enumerate(datasets):
    file_name = f"t_{i+1}"
    file_names.append(file_name)

print("Name of the file in the dataset", file_names)

#scaler = PowerTransformer()#StandardScaler()
#x = scaler.fit_transform(x)
print("Shape of spek after being scaled:", x.shape)

pca = PCA(n_components=3)
dataset_pca = pca.fit_transform(x)
print(type(dataset_pca))
print("After applying PCA, the shape of spek:", dataset_pca.shape)
# print(speak_pca)

first_pc = pca.components_[0]

# Print the number of wavenumbers in the first PC
num_wavenumbers_first_pc = len(first_pc)
print("Number of wavenumbers in the first PC:", num_wavenumbers_first_pc)

print('\n')


# # Reshape the datasets to (n_samples, n_features)
# r_datasets = x.reshape(16, -1)
# print("Print the shape of reshaped datasets", r_datasets.shape)

# # Reshape the datasets to (n_samples, n_features)
# reshaped_datasets = x.reshape(16, 49152*3)
# print("Print the shape of reshaped datasets", reshaped_datasets.shape)
# print('\n\n')

# append the dataset into a panda date frame
x_df = pd.DataFrame(dataset_pca)
# print(x_df.shape)

# # Sample a subset of columns
# sample_columns = x_df.sample(n=10, axis=1)  # Select 20 random columns
# print(sample_columns.head())

# plot the original dataset without k means clustering
plt.scatter(x_df[0],x_df[1])

# Plot the file names as labels for each data point
for i in range(len(dataset_pca)):
    plt.text(dataset_pca[i, 0], dataset_pca[i, 1], file_names[i], fontsize=8)

# Plot the data after applying PCA but before feeding into K_Means clustering
plt.show()


# Apply K-means clustering
# Set the number of clusters and the range of random states to try
num_clusters = 2
random_state_range = range(42)

# Initialize variables to store the best random state and the corresponding inertia
best_random_state = None
best_inertia = np.inf

# Iterate over different random states
for random_state in random_state_range:
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init = 'auto', max_iter=300, tol=0.0001,
                    verbose=1, random_state=random_state)
    kmeans.fit(dataset_pca)

    # Check if the current inertia is better than the previous best inertia
    if kmeans.inertia_ < best_inertia:
        best_inertia = kmeans.inertia_
        best_random_state = random_state


# Create the final KMeans model with the best random state
final_kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init = 'auto', max_iter=300, tol=0.0001,
                    verbose=1, random_state=best_random_state)
final_kmeans.fit(dataset_pca)

print("Status of K_Means", final_kmeans)
print("Best Random State:", best_random_state)
print("Best Inertia:", best_inertia)
print('\n\n')

# Get the final cluster labels and cluster centers
cluster_centers = final_kmeans.cluster_centers_
print("Centroid of K_Means", cluster_centers)
print('\n\n')

# Get the cluster labels
cluster_labels = final_kmeans.labels_
gt = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]
print("Print the label of cluster:         ", cluster_labels)
print("Print the original label of cluster: [0 0 1 1 1 0 1 0 1 1 0 1 1 1 1 0]") #[0 0 1 1 1 0 1 0 1 1 0 1 1 1 1 0]
print("0 is classified as soft texture and 1 is classified as hard texture")
print("Hit-miss:", cluster_labels==gt)


# print("Centroid of K_Means", kmeans.cluster_centers_)
# print('\n\n')
# # Print the best random state
# print("Best Random State:", best_random_state)
# print('\n\n')

# # Apply K-means clustering
# kmeans = KMeans(init='k-means++', n_init = 500, n_clusters=2)
# dataset_kmeans = kmeans.fit(dataset_pca )
# # print("?????", dataset_kmeans)
#
# print("Status of K_Means", kmeans)
# print('\n\n')
# print("Centroid of K_Means", kmeans.cluster_centers_)
# print('\n\n')


# # Get the cluster labels
# cluster_labels = kmeans.labels_
# print("Print the label of cluster:         ", cluster_labels)
# print("Print the original label of cluster: [0 0 1 1 1 0 1 1 1 0 0 1 1 0 1 1]")
# print("0 is classified as soft texture and 1 is classified as hard texture")

# plot the centroid of each cluster
plt.scatter(final_kmeans.cluster_centers_[:,0],final_kmeans.cluster_centers_[:,1],color='green',marker='*',label='centroid', s=200)


# Separate the data points based on cluster labels
cluster_0 = dataset_pca[cluster_labels == 0]
cluster_1 = dataset_pca[cluster_labels == 1]

# Create a scatter plot for each cluster
plt.scatter(cluster_0[:, 0], cluster_0[:, 1], c='red', label='Cluster 0')
plt.scatter(cluster_1[:, 0], cluster_1[:, 1], c='black', label='Cluster 1')


# Plot the file names as labels for each data point
for i in range(len(dataset_pca)):
    plt.text(dataset_pca[i, 0], dataset_pca[i, 1], file_names[i], fontsize=8)


# Set labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')

# Add a legend
plt.legend()

# Show the plot
plt.show()