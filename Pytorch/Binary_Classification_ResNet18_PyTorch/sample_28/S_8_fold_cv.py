import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np
import random

from resnet18_PyTorch import ResNet, BasicBlock
from TrainingData_PyTorch import train, validate
from DataLoad_PyTorch import save_plots, TrainDataset, TestDataset, predict_test_data

import torchvision.transforms as transforms

import csv
import matplotlib.pyplot as plt

import Training_Dataset_PyTorch
import Test_Dataset_PyTorch

from sklearn.model_selection import StratifiedKFold

from captum.attr import Saliency
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='scratch',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision']
)
args = vars(parser.parse_args())


# Set seed.
seed = 42
# seed = 180
# seed = 90
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)


# Learning and training parameters.
epochs = 100
batch_size = 1
learning_rate = 0.0001
n_splits = 8

# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define model based on the argument parser string.
model = ResNet(img_channels=441, num_layers=18, block=BasicBlock, num_classes=1).to(device)
# model = ResNet(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1).to(device)
print(model)
# plot_name = 'resnet_scratch'

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss function.
criterion = nn.BCEWithLogitsLoss()

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Initialize lists to store the last epoch's training and validation accuracies for each fold.
last_epoch_train_acc_list = []
last_epoch_valid_acc_list = []

# List to store the classification results
classification_results = []

x = np.array(Training_Dataset_PyTorch.datasets)
classes = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])

y = torch.tensor(classes, dtype=torch.float32)

# Convert NumPy arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.as_tensor(y).long()

test = np.array(Test_Dataset_PyTorch.testdatasets)
test_tensor = torch.tensor(test, dtype=torch.float32)

# Create the StratifiedKFold instance.
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define the transformations you want to apply to the images (if any).
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally.
    transforms.RandomVerticalFlip(),
])

# if __name__ == '__main__':
def s_8_fold():
    # Initialize an empty list to store AUC and Balanced Accuracy for each fold
    fold_results = []
    test_loaders = []
    valid_labels = []

    # Inside the for loop for each fold
    for fold, (train_index, valid_index) in enumerate(skf.split(x_tensor, y_tensor)):
    # for fold, (train_index, valid_index) in enumerate(loo.split(x_tensor, y_tensor)):
        print(f"  Train: index={train_index}")
        print(f"  Val:  index={valid_index}")

        plot_name = f'resnet_scratch_fold_{fold}'
        # Lists to keep track of losses and accuracies.
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        saliency_maps = []  # List to store saliency maps for each batch
        valid_samples = []  # List to store validation sample for each fold

        # Inside the validation loop, collect predictions and labels for each batch.
        valid_preds = []

        # Start the training.
        last_epoch = float('inf')  # Initialize with a large value
        epochs_without_improvement = 0

        # Get the train and validation data indices for this fold.
        train_indices, valid_indices = x_tensor[train_index], x_tensor[valid_index]
        train_labels, valid_labels = y_tensor[train_index], y_tensor[valid_index]

        print("print the shape of train_indices before scaling", train_indices.shape)

        flattened_train_indices = train_indices.reshape(train_indices.shape[0], -1)
        print("shape of flatten_train_indices", flattened_train_indices.shape)
        flattened_valid_indices = valid_indices.reshape(valid_indices.shape[0], -1)
        print("shape of flatten_valid_indices", flattened_valid_indices.shape)

        scaler = StandardScaler()
        scaler = scaler.fit(flattened_train_indices)
        scaled_train_indices = scaler.transform(flattened_train_indices)
        scaled_valid_indices = scaler.transform(flattened_valid_indices)

        # Reshape the flattened array back to the original shape
        train_indices = scaled_train_indices.reshape(train_indices.shape)
        print("print the shape of train_indices after scaling", train_indices.shape)
        valid_indices = scaled_valid_indices.reshape(valid_indices.shape)
        print("print the shape of valid_indices after scaling", valid_indices.shape)

        # Convert the NumPy array to a PyTorch tensor
        train_indices = torch.from_numpy(train_indices)
        valid_indices = torch.from_numpy(valid_indices)

        # Convert the PyTorch tensor to torch.cuda.FloatTensor
        train_indices = train_indices.to(device="cuda", dtype=torch.float32)
        print("print the shape of train_indices after converting to tensor", train_indices.shape)
        valid_indices = valid_indices.to(device="cuda", dtype=torch.float32)
        print("print the shape of train_indices after converting to tensor", valid_indices.shape)

        # Create the custom train and validation datasets for this fold.
        train_dataset = TrainDataset(train_indices, train_labels, transform=transform)
        valid_dataset = TrainDataset(valid_indices, valid_labels, transform=transform)

        # # Create the custom train and validation datasets for this fold.
        # train_dataset = TrainDataset(train_indices, train_labels, transform=None)
        # valid_dataset = TrainDataset(valid_indices, valid_labels, transform=None)

        # Create data loaders for this fold.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        print("the shape of test dataset", test.shape)  # 12, 49152, 441

        # reshape the testdataset to feed into the PCA
        reshaped_testdataset = np.reshape(test, (test.shape[0], -1))
        print("shape of reshaped_dataset:", reshaped_testdataset.shape)  # (12*49152), 441

        # # Retrieving Principal Components
        # pca_testdataset = pca.transform(reshaped_testdataset)
        # print("shape of reshaped_dataset after applying pca:", pca_testdataset.shape)  # 589824, 3

        # Applying standard scaling
        scaled_testdataset = scaler.transform(reshaped_testdataset)
        print("shape of test dataset after applying standard scaling", scaled_testdataset.shape)  # 12, 589824*441

        # Reshape back
        testdatasets = np.reshape(scaled_testdataset, (test.shape[0], test.shape[1], 441))
        print('Shape of whole test datasets', np.shape(testdatasets))  # 12, 49152, 441

        # Transpose the dataset to feed into Resnet 18
        testdatasets = np.transpose(testdatasets, (0, 2, 1))
        print('Shape of whole test datasets after being transposed', np.shape(testdatasets))  # 12, 441, 49152

        # reshape the dataset to feed into the ResNet 18
        testdatasets = np.reshape(testdatasets, (testdatasets.shape[0], testdatasets.shape[1], 128, 384))
        print("shape of final testdataset after being reshaped:", testdatasets.shape)  # 12, 441, 128, 384

        test_dataset = TestDataset(testdatasets, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders.append(test_loader)

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

        valid_labels = []
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
                    print("Raw logits of model confidence:", outputs)

                    outputs_sigmoid = torch.sigmoid(outputs).flatten().long()
                    print("Outputs after applying sigmoid:", outputs_sigmoid)

                    # Collect predictions probabilities and labels for each batch.
                    # valid_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
                    valid_preds.append(torch.sigmoid(outputs).cpu().numpy().squeeze())
                    print("print valid_preds", valid_preds)
                    print("print the shape of valid_preds:", np.shape(valid_preds))
                    valid_labels.append(labels.cpu().numpy().squeeze())
                    print("print valid_labels", valid_labels)
                    print("print the shape of valid_labels:", np.shape(valid_labels))

                    # Calculate saliency maps for the current batch
                    saliency = Saliency(model)
                    # saliency_map = saliency.attribute(samples, target=labels[0].item())
                    # saliency_map = saliency.attribute(samples, target=labels)  # , abs=False)
                    # saliency_map = saliency.attribute(samples, target=outputs_sigmoid)  # , abs=False)
                    saliency_map = saliency.attribute(samples, target=0)

                    saliency_maps.append(saliency_map)
                    print('length of saliency sample list', len(saliency_maps))
                    valid_samples.append(samples)

                    print('samples within for loop:', samples.shape)

        # Calculate AUC score
        auc_acc = roc_auc_score(valid_labels, valid_preds)
        print("AUC_Score:", auc_acc)

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
        fold_results.append((auc_acc, balanced_acc))

        # # Append the Balanced Accuracy score to the fold_results list
        # fold_results.append((balanced_acc))

        # print(f"Fold {fold + 1} - Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Fold {fold + 1} - AUC Score: {auc_acc:.4f}, Balanced Accuracy: {balanced_acc:.4f}")

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
            # writer.writerow(['Fold', 'Balanced Accuracy'])
            # for i, (bal_acc) in enumerate(fold_results):
            #     writer.writerow([i, bal_acc])

            writer.writerow(['Fold', 'AUC Score', 'Balanced Accuracy'])
            for i, (auc_acc, bal_acc) in enumerate(fold_results):
                writer.writerow([i, auc_acc, bal_acc])

        # print(f"Fold {fold + 1} - AUC Score: {auc_score:.4f}, Balanced Accuracy: {balanced_acc:.4f}")

        print('length of last epoch valid_samples:', len(valid_samples))

        # Plot sample images and their saliency maps
        for i in range(len(valid_samples)):
            print('shape of validation samples:', valid_samples[i][0].shape)
            ##plt.figure(figsize=(10, 5))
            ##plt.subplot(1, 2, 1)
            # sample = np.reshape(valid_samples[i][0].cpu().numpy(), (441, 128*384))
            sample = valid_samples[i][0].view(441, 128*384)
            # sample = valid_samples[i][0].view(1, 128 * 384)
            print('shape of sample after reshaping:', sample.shape)
            # sample = np.sum(sample, axis=0)
            sample = sample.sum(dim=0)
            # sample = np.reshape(sample, (128, 384))
            sample = sample.view(128, 384)
            print('shape of sample after reshaping:', sample.shape)
            sample = sample.cpu().numpy()
            plt.imshow(sample, cmap='jet')
            plt.title("Original Image")
            plt.axis('off')
            plt.show()

            print('length of saliency map', len(saliency_maps))
            print('shape of first element of saliency map', saliency_maps[i][0].shape)
            # saliency_sample = np.reshape(saliency_maps[i][0], (441, 128*384))
            saliency_sample = saliency_maps[i][0].view(441, 128*384)
            # saliency_sample = saliency_maps[i][0].view(1, 128 * 384)
            print('Shape of saliency sample', saliency_sample.shape)
            # saliency_sample = np.sum(saliency_sample, axis=0)
            saliency_sample = saliency_sample.sum(dim = 0)
            # saliency_sample = np.reshape(saliency_sample, (128, 384))
            saliency_sample = saliency_sample.view(128, 384)
            saliency_sample = saliency_sample.cpu().numpy()
            plt.imshow(saliency_sample, cmap='jet')
            plt.title("Saliency Map")
            plt.axis('off')
            plt.show()

        # Calculate False Positive Rate (FPR), True Positive Rate (TPR), and Thresholds
        fpr, tpr, thresholds = roc_curve(valid_labels, valid_preds)

        # Calculate Area Under the Curve (AUC)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    print('TRAINING COMPLETE')

def process_result():
    def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold", "7th Fold", "8th Fold"]
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
                "Accuracy scores in 8 Folds",
                last_epoch_train_acc_list,
                last_epoch_valid_acc_list)
