#Import necessary librariry
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from  torch import optim
from torchvision import transforms, utils, models
from torchvision.models import resnet18, ResNet18_Weights
import random
import csv  # Import the csv module.

from captum.attr import Saliency

from torchsummary import summary
from io import StringIO
import sys

from collections import OrderedDict
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# from resnet18_PyTorch import ResNet, BasicBlock
from TrainingData_16_PyTorch import train, validate
from DataLoad_16_PyTorch_TransferLearning import save_plots, get_data, predict_test_data

# Set the environment variable for deterministic behavior in CuBLAS
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8' depending on your system

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Learning and training parameters.
epochs = 100
batch_size = 1
learning_rate = 0.0001
n_splits = 6
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loaders, valid_loaders, test_loaders, test_loader = get_data(batch_size=batch_size, n_splits = n_splits)
# Define model based on the argument parser string.
# model =  models.resnet18(pretrained=True).to(device)
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# # Modify the first convolutional layer to accept 1 channel input
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify the first convolutional layer to accept 3 channel input
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Replace the last fully connected layer for binary classification
num_class = model.fc.in_features
model.fc = nn.Linear(num_class, 1)
model.to(device)

# print the resnet18 model
print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Specify the path where you want to save the summary
summary_path = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PyTorch_TransferLearning_ResNet18_model_summary.txt'

# Redirect the standard output to a file
with open(summary_path, 'w') as f:
    # Replace the standard output with the file
    sys.stdout = f
    # Use the summary function to get the model summary
    # summary(model, input_size=(1, 128, 384), device="cuda" if torch.cuda.is_available() else "cpu")
    summary(model, input_size=(3, 128, 384), device="cuda" if torch.cuda.is_available() else "cpu")
    # Reset the standard output
    sys.stdout = sys.__stdout__

# Optimizer.
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss function.
# criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Initialize lists to store the last epoch's training and validation accuracies for each fold.
last_epoch_train_acc_list = []
last_epoch_valid_acc_list = []

# List to store the classification results
classification_results = []

def samples_16_cv_transferlearning():
    # Initialize an empty list to store AUC and Balanced Accuracy for each fold
    fold_results = []

    # Inside the for loop for each fold
    for fold, (train_loader, valid_loader) in enumerate(zip(train_loaders, valid_loaders)):
        plot_name = f'resnet_scratch_fold_{fold}'
        # Lists to keep track of losses and accuracies.
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        saliency_maps = []  # List to store saliency maps for each batch
        valid_samples = []  # List to store validation sample for each fold

        # Inside the validation loop, collect predictions and labels for each batch.
        valid_preds = []
        valid_labels = []

        # Start the training.
        last_epoch = float('inf')  # Initialize with a large value
        epochs_without_improvement = 0
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
            train_epoch_loss, train_epoch_acc = train(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )
            valid_epoch_loss, valid_epoch_acc = validate(
                model,
                valid_loader,
                criterion,
                device
            )
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-' * 50)

            # # Early stopping check
            # best_train_loss = float('inf')  # Initialize with a large value
            # if train_epoch_loss < best_train_loss:
            #     best_train_loss = train_epoch_loss
            #     epochs_without_improvement = 0
            # else:
            #     epochs_without_improvement += 1
            #
            # if epochs_without_improvement >= patience:
            #     print(f"Early stopping: Training loss hasn't improved for {patience} epochs.")
            #     break  # Stop the training loop

            # last_epoch = epoch
            # print(last_epoch)

            # Early stopping check

            patience = 10  # Adjust this value as needed
            if epoch >= 50:  # Apply early stopping condition after 50 epochs
                if epoch == 50:
                    prev_train_loss = train_epoch_loss
                    print("previous training loss:", prev_train_loss)
                    print("Current training loss:", train_epoch_loss)
                elif epoch > 50:
                    print("previous training loss:", prev_train_loss)
                    curr_train_loss = train_epoch_loss
                    print("Current training loss:", train_epoch_loss)
                    if (curr_train_loss - prev_train_loss) < 1e-3:
                        epochs_without_improvement += 1
                    prev_train_loss = curr_train_loss
                else:
                    print("previous training loss:", prev_train_loss)
                    curr_train_loss = train_epoch_loss
                    print("Current training loss:", train_epoch_loss)
                    if (curr_train_loss - prev_train_loss) < 1e-3:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0
                    prev_train_loss = curr_train_loss

                if epochs_without_improvement >= patience:
                    print(f"Early stopping: Training loss hasn't improved for {patience} epochs.")
                    last_epoch = epoch
                    print("last epoch", last_epoch)
                    break  # Stop the training loop


        if epoch == epochs - 1 or epoch == last_epoch:
            print("last epoch", last_epoch)
            with ((torch.no_grad())):
                for samples, labels in valid_loader:
                    print("length of valid_loader", len(valid_loader))
                    samples, labels = samples.to(device), labels.to(device)
                    print("Shape of samples in valid loader", samples.shape)
                    print("shape of labels in valid loader", labels.shape)
                    print("print the labels of valid loader", labels)
                    print("Label:", labels[0].item())
                    print("Label:", labels.cpu().numpy())

                    outputs = model(samples)

                    # Collect predictions probabilities and labels for each batch.
                    # valid_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
                    valid_preds.append(torch.sigmoid(outputs).cpu().numpy().squeeze())
                    print("print valid_preds", valid_preds)
                    print("print the shape of valid_preds:", np.shape(valid_preds))
                    valid_labels.append(labels.cpu().numpy().squeeze())
                    print("print valid_labels", valid_labels)
                    print("print the label of valid_labels:", np.shape(valid_labels))

                    # Calculate saliency maps for the current batch
                    saliency = Saliency(model)
                    # saliency_map = saliency.attribute(samples, target=labels[0].item())
                    # saliency_map = saliency.attribute(samples, target=labels)  # , abs=False)
                    saliency_map = saliency.attribute(samples, target=None, abs=True)

                    saliency_maps.append(saliency_map)
                    print('length of saliency sample list', len(saliency_maps))
                    valid_samples.append(samples)

                    print('samples within for loop:', samples.shape)

        # Calculate AUC score
        auc_score = roc_auc_score(valid_labels, valid_preds)
        print("AUC_Score:", auc_score)

        for i in range(len(valid_preds)):
            if valid_preds[i] > 0.5:
                valid_preds[i] = 1
            else:
                valid_preds[i] = 0

        # Calculate Balanced Accuracy Score
        # balanced_acc = balanced_accuracy_score(valid_labels, (np.round(valid_preds)))
        balanced_acc = balanced_accuracy_score(valid_labels, valid_preds)
        print("Balance_Score:", balanced_acc)

        # Append the AUC score and Balanced Accuracy score to the fold_results list
        fold_results.append((auc_score, balanced_acc))
        # fold_results.append(balanced_acc)

        print(f"Fold {fold + 1} - AUC Score: {auc_score:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
        # print(f"Fold {fold + 1} - Balanced Accuracy: {balanced_acc:.4f}")

        print('length of valid samples:', len(valid_samples))
        print(valid_samples[0].shape)
        print('length of saliency sample list', len(saliency_maps))

        # Track the accuracies for each epoch.
        last_epoch_train_acc = train_acc[-1]
        print("Print last epoch training accuracy", last_epoch_train_acc)
        last_epoch_valid_acc = valid_acc[-1]
        print("Print last epoch validation accuracy", last_epoch_valid_acc)

        # Store the last epoch's training and validation accuracies for this fold.
        last_epoch_train_acc_list.append(last_epoch_train_acc)
        last_epoch_valid_acc_list.append(last_epoch_valid_acc)

        # Save the loss and accuracy plots.
        save_plots(
            train_acc,
            valid_acc,
            train_loss,
            valid_loss,
            name=plot_name
        )

        # Save the last epoch's training and validation accuracy to a CSV file.
        with open(f'/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/accuracy_fold_{fold}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Training Accuracy', 'Validation Accuracy'])
            for epoch, train_acc_epoch, valid_acc_epoch in zip(range(1, epochs + 1), train_acc, valid_acc):
                writer.writerow([epoch, train_acc_epoch, valid_acc_epoch])


        with open(f'/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/fold_{fold}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fold', 'AUC Score', 'Balanced Accuracy'])
            for i, (auc, bal_acc) in enumerate(fold_results):
                writer.writerow([i, auc, bal_acc])
            # for i, (bal_acc) in enumerate(fold_results):
            #     writer.writerow([i, bal_acc])

        # print(f"Fold {fold + 1} - AUC Score: {auc_score:.4f}, Balanced Accuracy: {balanced_acc:.4f}")

        print('length of last epoch valid_samples:', len(valid_samples))

        # Plot sample images and their saliency maps
        for i in range(len(valid_samples)):
            print('shape of validation samples:', valid_samples[i][0].shape)
            ##plt.figure(figsize=(10, 5))
            ##plt.subplot(1, 2, 1)
            # sample = np.reshape(valid_samples[i][0].cpu().numpy(), (441, 128*384))
            sample = valid_samples[i][0].view(3, 128 * 384)
            # sample = valid_samples[i][0].view(1, 128 * 384)
            print('shape of sample after reshaping:', sample.shape)
            # sample = np.sum(sample, axis=0)
            sample = sample.sum(dim=0)
            # sample = np.reshape(sample, (128, 384))
            sample = sample.view(128, 384)
            print('shape of sample after reshaping:', sample.shape)
            sample = sample.cpu().numpy()
            # plt.imshow(sample, cmap='jet')
            # plt.title("Original Image")
            # plt.axis('off')
            # plt.show()

            print('length of saliency map', len(saliency_maps))
            print('shape of first element of saliency map', saliency_maps[i][0].shape)

            # saliency_sample = np.reshape(saliency_maps[i][0], (441, 128*384))
            saliency_sample = saliency_maps[i][0].view(3, 128 * 384)
            # saliency_sample = saliency_maps[i][0].view(1, 128 * 384)
            print('Shape of saliency sample', saliency_sample.shape)
            # saliency_sample = np.sum(saliency_sample, axis=0)
            saliency_sample = saliency_sample.sum(dim = 0)
            # saliency_sample = np.reshape(saliency_sample, (128, 384))
            saliency_sample = saliency_sample.view(128, 384)
            ##plt.subplot(1, 2, 2)
            saliency_sample = saliency_sample.cpu().numpy()
            # plt.imshow(saliency_sample, cmap='jet')
            # # plt.imshow(sample, cmap='hot')
            # plt.title("Saliency Map")
            # plt.axis('off')
            # plt.show()

    print('TRAINING COMPLETE')

def process_result():
    def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold"]
        # labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
        # labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 100)
        plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/accuracy_per_fold_plot.png')


    # Plot Accuracy Result
    model_name = "Bar Chart"
    plot_result(model_name,
                "Accuracy",
                "Accuracy scores in 6 Folds",
                last_epoch_train_acc_list,
                last_epoch_valid_acc_list)
