import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import random

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.preprocessing import StandardScaler

import Training_Dataset_LR_PyTorch
import Test_Dataset_LR_PyTorch

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


# Define your custom dataset class.
class TrainDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].numpy()  # Convert Tensor to NumPy ndarray
        # Convert the 4-dimensional tensor to a 3-dimensional NumPy array with 3 channels
        # sample = np.transpose(sample, (1, 2, 0))
        label = self.labels[idx]

        # if self.transform:
        #     sample = self.transform(sample)

        return sample, label


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


x = np.array(Training_Dataset_LR_PyTorch.datasets)
print("print x:", x.shape)

# Generate labels for each image
classes = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
# classes = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
y = torch.tensor(classes, dtype=torch.float32)

# test = np.array(Test_Dataset_LR_PyTorch.testdatasets)
# test_tensor = torch.tensor(test, dtype=torch.float32)

# Convert NumPy arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
print("print x_tensor:", x_tensor.shape)
y_tensor = torch.as_tensor(y).long()

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# Load the data and split it into training and validation sets
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# def get_data(batch_size=1, n_splits=4):
def get_data(batch_size=1, n_splits=6):
    # Define the path to your custom IR Spectral dataset.
    root_dir = 'path_to_your_custom_dataset_folder'

    # Define the transformations you want to apply to the images (if any).
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally.
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image.
        # Add more transformations as needed for your specific dataset.
    ])

    # Create the StratifiedKFold instance.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store the data loaders for each fold.
    train_loaders = []
    valid_loaders = []
    test_loaders = []

    # test_dataset = TestDataset(test, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # test_loaders.append(test_loader)

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

    train_indices = []
    train_labels = []
    valid_indices = []
    valid_labels = []

    # for train_index, valid_index in kf.split(x_tensor, y_tensor):
    # for train_index, valid_index in fold_indices:
    for train_index, valid_index in skf.split(x_tensor, y_tensor):
        print(f"  Train: index={train_index}")
        print(f"  Val:  index={valid_index}")

        # Get the train and validation data indices for this fold.
        train_ind, valid_ind = x_tensor[train_index], x_tensor[valid_index]
        print("print train_indices:", train_ind.shape)
        train_lab, valid_lab = y_tensor[train_index], y_tensor[valid_index]
        print("print train_labels:", train_lab.shape)

        scaler = StandardScaler()
        scaler = scaler.fit(train_ind)
        scaled_train_indices = scaler.transform(train_ind)
        scaled_valid_indices = scaler.transform(valid_ind)

        train_indices.append(scaled_train_indices)
        train_labels.append(train_lab)
        valid_indices.append(scaled_valid_indices)
        valid_labels.append(valid_lab)

        # # Create the custom train and validation datasets for this fold.
        # train_dataset = TrainDataset(train_indices, train_labels, transform=transform)
        # valid_dataset = TrainDataset(valid_indices, valid_labels, transform=transform)

    test = np.array(Test_Dataset_LR_PyTorch.testdatasets)
    test = scaler.transform(test)

    test_dataset = TestDataset(test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loaders.append(test_loader)

    return train_indices, train_labels, valid_indices, valid_labels, test_loader, test_loaders


def predict_test_data(model, test_loader, device):
    """
    Function to predict the test data using the trained model.
    """
    test_preds_list = []  # Initialize an empty list to store test predictions.
    outputs_prob_class_list = []

    with torch.no_grad():
        for samples in test_loader:
            samples = samples.to(device)

            # Convert the input tensor to the same data type as the model's weight tensor.
            samples = samples.float()

            # Make prediction probabilities on the test samples.
            outputs = model.predict_proba(samples.to('cpu'))
            print("model confidence on test data:", outputs)
            test_preds_list .append(outputs)

            # make label predictions on the test samples.
            outputs_prob_class = model.predict(samples.to('cpu'))
            print("Class probability on test data:", outputs_prob_class)
            outputs_prob_class_list.append(outputs_prob_class)

    # Concatenate the test predictions from all batches into a single NumPy array.
    test_preds_array = np.concatenate(test_preds_list, axis=0)
    outputs_prob_class_array = np.concatenate(outputs_prob_class_list, axis=0)

    return test_preds_array, outputs_prob_class_array


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