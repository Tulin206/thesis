import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import numpy as np
import random
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.preprocessing import StandardScaler

import Training_Dataset_16_Mob_V1_PyTorch
import Test_Dataset_Mob_V1_PyTorch

# Set seed.
# seed = 180
# seed = 90
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# store the list of augmented sample
aug_samp = []

# Define your custom dataset class.
class TrainDataset(Dataset):
    def __init__(self, data, labels, transform=None, fold_idx=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.fold_idx = fold_idx  # Store the fold index
        # Directory to save augmented images
        self.save_dir = f'/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/Fold_{fold_idx}/'

        # Create the save directory if it doesn't exist
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_sample = self.data[idx]
        # sample = self.data[idx].numpy()  # Convert Tensor to NumPy ndarray
        label = self.labels[idx]

        # Make a copy of the original image
        original_sample = raw_sample.clone()
        print(f"shape of original sample: {original_sample.shape}")

        # visualize original image
        org_img = raw_sample.cpu().numpy()
        # org_img = np.reshape(org_img, (1, 128 * 384))
        org_img = np.reshape(org_img, (441, 128*384))
        org_img = np.sum(org_img, axis=0)
        org_img = np.reshape(org_img, (128, 384))
        # plt.imshow(org_img, cmap='jet')  # Rearrange dimensions for visualization
        # plt.title(f"Original Sample - Label: {label}")
        # plt.show()

        if self.transform:
            # Apply transformations
            transformed_sample = self.transform(raw_sample)
            # Check the shape of the sample
            print(f"Sample shape after transforming: {transformed_sample.shape}")

            # Plot the augmented image using matplotlib
            augmented_sample = transformed_sample.cpu().numpy()
            # augmented_sample = augmented_sample.view(441, 128 * 384)
            augmented_sample = np.reshape(augmented_sample, (441, 128 * 384))
            # augmented_sample = np.reshape(augmented_sample, (1, 128 * 384))
            # augmented_sample = augmented_sample.sum(dim=0)
            augmented_sample = np.sum(augmented_sample, axis=0)
            # augmented_sample = augmented_sample.view(128, 384)
            augmented_sample = np.reshape(augmented_sample, (128, 384))
            print("shape of sample after having augmented:", augmented_sample.shape)
            aug_samp.append(augmented_sample)
            print("length of aug_samp list:", len(aug_samp))
            # plt.imshow(augmented_sample, cmap='jet')  # Rearrange dimensions for visualization
            # plt.title(f"Augmented Sample - Label: {label}")
            # plt.show()

            # # Save augmented image (optional)
            # if self.save_dir:
            #     for i, augmented_sample in enumerate(aug_samp):
            #         filename = f"augmented_image_{idx}_{i}.png"
            #         save_path = os.path.join(self.save_dir, filename)
            #         plt.imsave(save_path, augmented_sample)  # Save as image file

        if self.transform:
            # concatenate original image with augmented image
            sample = torch.stack((original_sample, transformed_sample), dim=0)

            # Check the shape of the sample
            print(f"Sample shape: {sample.shape}")

            num_images_in_sample = sample.shape[0]
            print(f"Total number of images in sample: {num_images_in_sample}")
            print("shape of first sample:", sample[0].shape)

            # Check the shape of the sample
            print(f"Sample shape: {sample.shape}")

            for single_sample in sample:
                return single_sample, label

        else:
            return raw_sample, label

class TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert the 4-dimensional tensor to a 3-dimensional NumPy array with 3 channels
        # sample = np.transpose(sample, (1, 2, 0))

        # if self.transform:
        #     sample = self.transform(sample)

        return sample

# creating array of input images
x = np.array(Training_Dataset_16_Mob_V1_PyTorch.datasets)

# Generate true labels for 16 samples
classes = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
y = torch.tensor(classes, dtype=torch.float32)

test = np.array(Test_Dataset_Mob_V1_PyTorch.testdatasets)
test_tensor = torch.tensor(test, dtype=torch.float32)

# Convert NumPy arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.as_tensor(y).long()

def calculate_class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (2 * class_counts)

    # # Convert class weights to PyTorch tensor
    # class_weights_tensor = torch.FloatTensor(class_weights)
    #
    # # Extract class weights based on true class labels
    # batch_class_weights = class_weights_tensor[labels]
    # return batch_class_weights
    return torch.FloatTensor(class_weights)

# Calculate class weights
class_counts = np.bincount(classes)
total_samples = len(classes)
class_weights = total_samples / (2 * class_counts)
print("print the shape of class_weights", class_weights.shape)

# Access the class weights for each class
weight_class_0 = class_weights[0]
weight_class_1 = class_weights[1]

print("Weight for class 0 (Soft):", weight_class_0)
print("Weight for class 1 (Hard):", weight_class_1)

# Convert class weights to PyTorch tensor
class_weights_tensor = torch.FloatTensor(class_weights)

# Extract class weights based on true class labels
batch_class_weights = class_weights_tensor[classes]
print(batch_class_weights)

def predict_test_data(model, test_loader, device):
    """
    Function to predict the test data using the trained model.
    """
    model.eval()  # Set the model to evaluation mode.
    test_preds_list = []  # Initialize an empty list to store test predictions.

    with torch.no_grad():
        for samples in test_loader:
            samples = samples.to(device)

            # Convert the input tensor to the same data type as the model's weight tensor.
            samples = samples.float()

            # Make predictions on the test samples.
            outputs = model(samples)
            # test_preds = torch.softmax(outputs, dim=1).cpu().numpy()
            # test_preds_list.append(test_preds)
            test_preds_list .append(outputs.cpu().numpy())

    # Concatenate the test predictions from all batches into a single NumPy array.
    test_preds_array = np.concatenate(test_preds_list, axis=0)

    return test_preds_array

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# # Load the trained model (replace 'path_to_saved_model.pth' with the actual path to your saved model file).
# model = torch.load('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/saved_model.pth')
# model = model.to(device)


# Inside the for loop for each fold
def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/', f'{name}_accuracy.png'))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/', f'{name}_loss.png'))