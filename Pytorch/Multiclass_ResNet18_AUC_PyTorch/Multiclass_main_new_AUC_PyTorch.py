import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import argparse
import numpy as np
import random

from Multiclass_resnet18_AUC_PyTorch import ResNet, BasicBlock
from Multiclass_TrainingData_AUC_PyTorch import train, validate
from Multiclass_DataLoad_AUC_PyTorch import save_plots, get_data, predict_test_data

import csv  # Import the csv module.
import matplotlib.pyplot as plt

from captum.attr import Saliency

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
n_splits = 2
# n_splits = 4
# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loaders, valid_loaders, test_loaders, test_loader = get_data(batch_size=batch_size, n_splits = n_splits)

# Define model based on the argument parser string.
model = ResNet(img_channels=441, num_layers=18, block=BasicBlock, num_classes=3).to(device)
# model = ResNet(img_channels=1, num_layers=18, block=BasicBlock, num_classes=3).to(device)
print(model)
# plot_name = 'resnet_scratch'

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = MobileNetV1(ch_in=441, n_classes=1).to(device)
# summary(model, input_size=(441, 128, 384), device=device)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss function.
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()


from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Initialize lists to store the last epoch's training and validation accuracies for each fold.
last_epoch_train_acc_list = []
last_epoch_valid_acc_list = []

# List to store the classification results
classification_results = []

if __name__ == '__main__':

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
        valid_preds_AUC = []
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
                    print("length of valid_loader", len(valid_loader))  # length of valid_loader 3
                    samples, labels = samples.to(device), labels.to(device)
                    print("Shape of samples in valid loader", samples.shape)  # torch.Size([1, 441, 128, 384])
                    print("shape of labels in valid loader", labels.shape)  # torch.Size([1])
                    print("print the labels of valid loader", labels)  # tensor([0], device='cuda:0')
                    # print("Label:", labels[0].item())
                    print("Label:", labels.cpu().numpy())   # Label: [0]

                    outputs = model(samples)
                    print("print outputs", outputs)  # tensor([[-2.0953,  3.7289, -1.2528]], device='cuda:0')
                    print("shape of outputs", outputs.shape)  # torch.Size([1, 3])

                    outputs = torch.softmax(outputs, dim=1)
                    print("print outputs after applying softmax", outputs)  # tensor([[0.0029, 0.9903, 0.0068]], device='cuda:0')
                    print("shape of outputs after applying softmax", outputs.shape)  # torch.Size([1, 3])

                    # For feeding into the ROC_AUC
                    valid_preds_AUC.append(outputs.cpu().numpy().squeeze())
                    print("shape of valid_preds_AUC:", np.shape(valid_preds_AUC))   # before sqeezing it was shape of valid_preds_AUC: (8, 1, 3)

                    outputs = outputs.cpu().numpy().squeeze()
                    print("print outputs", outputs)  # [0.0029265  0.9902777  0.00679573]

                    outputs = np.max(outputs)
                    # outputs, _ = torch.max(torch.tensor(outputs, dtype=torch.float32), dim=1)
                    print("maximum probability of outputs", outputs)  # 0.9902777

                    # Collect predictions probabilities for each batch.
                    valid_preds.append(outputs)
                    # valid_preds.append(torch.softmax(outputs, dim=1).cpu().numpy().squeeze())
                    # valid_preds.append(torch.sigmoid(outputs).cpu().numpy().squeeze())
                    print("print valid_preds", valid_preds)  # print valid_preds [0.9902777]
                    print("shape of valid_preds:", np.shape(valid_preds))  # shape of valid_preds: (1,)

                    # Collect true labels for each batch
                    # valid_labels.append(labels.cpu().numpy())
                    valid_labels.append(labels.cpu().numpy().squeeze())
                    print("print valid_labels", valid_labels)  # print valid_labels [array(0)]
                    print("shape of valid_labels:", np.shape(valid_labels))  # shape of valid_labels: (1,)

                    # Calculate saliency maps for the current batch
                    saliency = Saliency(model)
                    # saliency_map = saliency.attribute(samples, target=labels[0].item())
                    # saliency_map = saliency.attribute(samples, target=labels)  # , abs=False)
                    saliency_map = saliency.attribute(samples, target=1)  # , abs=False)

                    saliency_maps.append(saliency_map)
                    print('length of saliency sample list', len(saliency_maps))   # length of saliency sample list 1
                    valid_samples.append(samples)

                    print('samples within for loop:', samples.shape)  # torch.Size([1, 441, 128, 384])
                    print("////////// end of loop for valid loader //////////")


        print("List of valid_labels:", valid_labels)  # [array(0), array(1), array(1), array(0), array(1), array(2), array(1), array(1)]
        print("Shape of the list for valid_labels", np.shape(valid_labels))   # Shape of the list for valid_labels (8,)
        print("List of valid_preds:", valid_preds)    # [0.7917804, 0.99427575, 0.70360625, 0.9895419, 0.9601784, 0.82928884, 0.8324819, 0.93372065]
        print("Shape of the list for valid_preds", np.shape(valid_preds))  # Shape of the list for valid_preds (8,)
        print("Shape of the list for valid_preds_AUC", np.shape(valid_preds_AUC))  # (8, 3)
        print("List of print valid_preds_AUC", valid_preds_AUC)        # [array([0.13856873, 0.7917804 , 0.06965081], dtype=float32), array([5.6230631e-03, 9.9427575e-01, 1.0125215e-04], dtype=float32), array([0.16071105, 0.70360625, 0.13568273], dtype=float32), array([1.0149495e-02, 9.8954189e-01, 3.0861390e-04], dtype=float32), array([0.03576072, 0.9601784 , 0.00406089], dtype=float32), array([0.09512118, 0.82928884, 0.07558997], dtype=float32), array([0.09891432, 0.8324819 , 0.06860374], dtype=float32), array([0.0375405 , 0.93372065, 0.02873884], dtype=float32)]
        # print("valid_labels after applying arg-max:", np.argmax(valid_labels, axis =1))
        # print("valid_preds after applying arg-max:", np.argmax(valid_preds, axis=1))

        # Calculate Balanced Accuracy Score
        valid_preds = np.round(valid_preds)
        print("print valid_preds after rounding up:", valid_preds)  # [1. 1. 1. 1. 1. 1. 1. 1.]
        balanced_acc = balanced_accuracy_score(valid_labels, (np.round(valid_preds)))
        # balanced_acc = balanced_accuracy_score(valid_labels, np.argmax(valid_preds, axis=1))  # find the class label with the highest predicted probability for each sample along the specified axis (axis 1).
        # balanced_acc = balanced_accuracy_score(np.argmax(valid_labels, axis =1), np.argmax(valid_preds, axis = 1))
        print("Balance_Score:", balanced_acc)

        auc_score = roc_auc_score(valid_labels, valid_preds_AUC, multi_class='ovr') # shape of valid_labels: (8,) & shape of valid_preds_AUC: (8, 3)
        print("ROC AUC Score:", auc_score)

        # Append the AUC score and Balanced Accuracy score to the fold_results list
        fold_results.append((auc_score, balanced_acc))
        # fold_results.append(balanced_acc)

        print(f"Fold {fold + 1} - AUC Score: {auc_score:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
        # print(f"Fold {fold + 1} - Balanced Accuracy: {balanced_acc:.4f}")

        print('length of valid samples:', len(valid_samples))  # length of valid samples: 3
        print("shape of valid_samples", valid_samples[0].shape)  # torch.Size([1, 441, 128, 384])
        print('length of saliency sample list', len(saliency_maps))  # length of saliency sample list 3

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
            # writer.writerow(['Fold', 'AUC Score', 'Balanced Accuracy'])
            # for i, (auc, bal_acc) in enumerate(fold_results):
            #     writer.writerow([i, auc, bal_acc])
            writer.writerow(['Fold', 'Balanced Accuracy'])
            for i, (bal_acc) in enumerate(fold_results):
                writer.writerow([i, bal_acc])
        # print(f"Fold {fold + 1} - AUC Score: {auc_score:.4f}, Balanced Accuracy: {balanced_acc:.4f}")

        print('length of last epoch valid_samples:', len(valid_samples))  # length of last epoch valid_samples: 3

        # Plot sample images and their saliency maps
        for i in range(len(valid_samples)):
            print('shape of validation samples:', valid_samples[i][0].shape)  # torch.Size([441, 128, 384])
            ##plt.figure(figsize=(10, 5))
            ##plt.subplot(1, 2, 1)
            # sample = np.reshape(valid_samples[i][0].cpu().numpy(), (441, 128*384))
            sample = valid_samples[i][0].view(441, 128*384)
            # sample = valid_samples[i][0].view(1, 128 * 384)
            print('shape of sample after reshaping:', sample.shape) # torch.Size([441, 49152])
            # sample = np.sum(sample, axis=0)
            sample = sample.sum(dim=0)
            # sample = np.reshape(sample, (128, 384))
            sample = sample.view(128, 384)
            print('shape of sample after reshaping:', sample.shape) # torch.Size([128, 384])
            sample = sample.cpu().numpy()
            plt.imshow(sample, cmap='jet')
            plt.title("Original Image")
            plt.axis('off')
            plt.show()

            print('length of saliency map', len(saliency_maps))  # length of saliency map 3
            print('shape of first element of saliency map', saliency_maps[i][0].shape)  # torch.Size([441, 128, 384])

            # saliency_sample = np.reshape(saliency_maps[i][0], (441, 128*384))
            saliency_sample = saliency_maps[i][0].view(441, 128*384)
            # saliency_sample = saliency_maps[i][0].view(1, 128 * 384)
            print('Shape of saliency sample', saliency_sample.shape)   # torch.Size([441, 49152])
            # saliency_sample = np.sum(saliency_sample, axis=0)
            saliency_sample = saliency_sample.sum(dim = 0)
            # saliency_sample = np.reshape(saliency_sample, (128, 384))
            saliency_sample = saliency_sample.view(128, 384)
            ##plt.subplot(1, 2, 2)
            saliency_sample = saliency_sample.cpu().numpy()
            plt.imshow(saliency_sample, cmap='jet')
            plt.title("Saliency Map")
            plt.axis('off')
            plt.show()

    print('TRAINING COMPLETE')


def plot_result(x_label, y_label, plot_title, train_data, val_data):
    # Set size of plot
    plt.figure(figsize=(12, 6))
    # labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold"]
    labels = ["1st Fold", "2nd Fold"]
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
            "Accuracy scores in 5 Folds",
            last_epoch_train_acc_list ,
            last_epoch_valid_acc_list)


# Make predictions on the test dataset.
test_preds = predict_test_data(model, test_loader, device)
print('probability labels of test data\n', test_preds)
test_preds = np.max(test_preds, axis=1)  # Get maximum predicted class for each sample
print('labels of test data\n', test_preds)

# for i in preds:
for i in np.nditer(test_preds):
        print(i)
        if i <= 0.5:
            # print("The image classified is soft texture")
            classification = "The image is classified as soft texture"
            # writer.writerow([message])
        elif i <= 0.7:
            # print("The image classified is soft texture")
            classification = "The image is classified as Medium texture"
            # writer.writerow([message])
        else:
            # print("The image classified is hard texture")
            classification = "The image is classified as hard texture"
            # writer.writerow([message])

        classification_results.append(classification)  # Store the classification in the list

# Write the classification results to the CSV file
with open('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/Texture_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Classification'])

    # Write the classification results
    for result in classification_results:
        writer.writerow([result])

print("CSV file has been created successfully.")

saliency_test_maps = []  # List to store saliency maps for each batch
test_samples = []  # List to store validation sample for each fold

test_pred = torch.tensor(test_preds, dtype=torch.float32, device=device)
print("Maximum labels of test data as a tensor\n", test_pred)
print("Size of the tensor where the labels of the test data are stored", test_pred.shape)

# # Convert test_preds to binary values based on threshold
# test_pred_binary = np.where(test_preds < 0.5, 0, 1)

# # Define your threshold values
# threshold_low = 0.5
# threshold_medium = 0.6
#
# # Convert test_preds to binary values based on thresholds
# test_pred_binary = torch.where(test_pred <= threshold_low, 0,  # If <= 0.5, set to 0
#                                torch.where(test_pred <= threshold_medium, 2,  # If <= 0.6, set to 2
#                                            1))  # If > 0.6, set to 1
#
# test_pred_tensor = torch.as_tensor(test_pred_binary).long()
# print("Shape of test_pred_tensor", test_pred_tensor.shape)  # torch.Size([12])
# print("print test_pred_tensor", test_pred_tensor)  # tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')

for test_loader in test_loaders:
    with ((torch.no_grad())):
        for i, samples in enumerate(test_loader):
            samples = samples.to(device)

            # Convert the input samples to float data type
            samples = samples.float()
            outputs = model(samples)

            # # Calculate the class probabilities for the current sample
            # test_pred_sample = test_pred_tensor[i].item()

            # Calculate saliency maps for the current batch
            saliency = Saliency(model)
            # saliency_map = saliency.attribute(samples, target=test_pred_sample)
            saliency_map = saliency.attribute(samples, target=1)

            saliency_test_maps.append(saliency_map)
            test_samples.append(samples)

# Plot sample images and their saliency maps
for i in range(len(test_samples)):
    print('shape of validation samples:', test_samples[i][0].shape)
    ##plt.figure(figsize=(10, 5))
    ##plt.subplot(1, 2, 1)
    # sample = np.reshape(valid_samples[i][0].cpu().numpy(), (441, 128*384))
    sample = test_samples[i][0].view(441, 128 * 384)
    # sample = test_samples[i][0].view(1, 128 * 384)
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

    print('length of saliency map', len(saliency_test_maps))
    print('shape of first element of saliency map', saliency_test_maps[i][0].shape)

    # saliency_sample = np.reshape(saliency_maps[i][0], (441, 128*384))
    saliency_sample = saliency_test_maps[i][0].view(441, 128 * 384)
    # saliency_sample = saliency_test_maps[i][0].view(1, 128 * 384)
    print('Shape of saliency sample', saliency_sample.shape)
    # saliency_sample = np.sum(saliency_sample, axis=0)
    saliency_sample = saliency_sample.sum(dim=0)
    # saliency_sample = np.reshape(saliency_sample, (128, 384))
    saliency_sample = saliency_sample.view(128, 384)
    ##plt.subplot(1, 2, 2)
    saliency_sample = saliency_sample.cpu().numpy()
    plt.imshow(saliency_sample, cmap='jet')
    plt.title("Saliency Map")
    plt.axis('off')
    plt.show()
