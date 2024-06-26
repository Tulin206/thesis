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

import Training_Dataset_SR_PyTorch
import Test_Dataset_SR_PyTorch

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
        # # Without Augmentation
        # sample = self.data[idx]
        # label = self.labels[idx]
        #
        # # if self.transform:
        # #     sample = self.transform(sample)
        #
        # return sample, label

        # With Augmentation
        raw_sample = self.data[idx]
        # sample = self.data[idx].numpy()  # Convert Tensor to NumPy ndarray
        label = self.labels[idx]

        # Make a copy of the original image
        original_sample = raw_sample.clone()
        print(f"shape of original sample: {original_sample.shape}")

        # visualize original image
        org_img = raw_sample.cpu().numpy()
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
x = np.array(Training_Dataset_SR_PyTorch.datasets)

# Generate labels for each image
classes = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
# classes = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
y = torch.tensor(classes, dtype=torch.float32)

test = np.array(Test_Dataset_SR_PyTorch.testdatasets)
test_tensor = torch.tensor(test, dtype=torch.float32)

# Convert NumPy arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.as_tensor(y).long()

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


# def get_data(batch_size=1, n_splits=4):
def get_data(batch_size=1, n_splits=6):

    # Define the transformations you want to apply to the images (if any).
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally.
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=66),
        # transforms.ColorJitter(),
        # Add more transformations as needed for your specific dataset.
    ])

    # # Create the custom dataset instances and visualise augmented image
    # dataset_train = TrainDataset(x_tensor, y_tensor, transform=transform)
    # dataset_train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    # Create the StratifiedKFold instance.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store the data loaders for each fold.
    train_loaders = []
    valid_loaders = []
    test_loaders = []

    test_dataset = TestDataset(test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loaders.append(test_loader)

    # # Define your fold indices manually
    # fold_indices = [
    #     ([1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 4, 12], [0, 3]),
    #     ([0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15], [6, 7, 11]),
    #     ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [14, 15]),
    #     ([0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15], [3, 10, 13]),
    #     ([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15], [5, 9]),
    #     # Define indices for the other folds similarly
    #     # ...
    # ]


    # # Define the indices of the datasets you want to exclude from validation.
    # exclude_indices = [2, 4, 5, 8, 12]

    # for train_index, valid_index in kf.split(x_tensor, y_tensor):
    # for train_index, valid_index in fold_indices:
    for train_index, valid_index in skf.split(x_tensor, y_tensor):

        # # Get the validation data indices for this fold.
        # valid_index = np.setdiff1d(valid_index, exclude_indices)

        print(f"  Train: index={train_index}")
        print(f"  Val:  index={valid_index}")

        # Get the train and validation data indices for this fold.
        train_indices, valid_indices = x_tensor[train_index], x_tensor[valid_index]
        train_labels, valid_labels = y_tensor[train_index], y_tensor[valid_index]

        # Create the custom train and validation datasets for this fold.
        train_dataset = TrainDataset(train_indices, train_labels, transform=transform)
        valid_dataset = TrainDataset(valid_indices, valid_labels, transform=transform)

        # Create data loaders for this fold.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Append the data loaders to the lists.
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)

    # # Create data loaders.
    # train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    return train_loaders, valid_loaders, test_loaders, test_loader


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
            test_preds_list .append(torch.sigmoid(outputs).cpu().numpy())

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