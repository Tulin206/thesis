import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import argparse
import numpy as np
import random

from sklearn.linear_model import LogisticRegression
# from Logistic_Regression_PyTorch import LogisticRegression
from DataLoad_LR_PyTorch import save_plots, get_data, predict_test_data
# from DataLoad_single_fold_PyTorch import save_plots, get_data, predict_test_data

from sklearn.preprocessing import StandardScaler
import csv  # Import the csv module.
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

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
n_splits = 6
# n_splits = 4
# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_indices, train_labels, valid_indices, valid_labels, test_loader, test_loaders = get_data(batch_size=batch_size, n_splits = n_splits)

# Initialize logistic regression classifier (model)
model = LogisticRegression(random_state=42) #, verbose=1)
print(model)


from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Initialize lists to store the last epoch's training and validation accuracies for each fold.
last_epoch_train_acc_list = []
last_epoch_valid_acc_list = []

# List to store the classification results
classification_results = []

if __name__ == '__main__':

    # Initialize an empty list to store AUC and Balanced Accuracy for each fold
    fold_results = []
    train_accuracy = []
    valid_accuracy = []
    train_loss = []
    valid_loss = []

    # Inside the for loop for each fold
    for fold, (train_indices, train_labels, valid_indices, valid_labels) in enumerate(zip(train_indices, train_labels, valid_indices, valid_labels)):
        plot_name = f'Logistic_Regression_fold_{fold}'

        # Inside the validation loop, collect predictions and labels for each batch.
        valid_preds_list = []
        valid_labels_list = []

        # Start the training.
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
            print("print shape of train_indices:", train_indices.shape)
            print("print shape of train_labels:", train_labels.shape)
            print("print train_labels:", train_labels)
            print("print shape of valid_indices:", valid_indices.shape)
            print("print shape of valid_labels:", valid_labels.shape)
            print("print valid_labels:", valid_labels)
            model.fit(train_indices, train_labels)

            # Make predictions on the training set
            train_preds = model.predict(train_indices)
            print("Probability label on training data:", train_preds)

            # Calculate training accuracy
            # train_epoch_accuracy = 100. * (accuracy_score(train_labels, train_preds))
            train_epoch_accuracy = 100. * (model.score(train_indices, train_labels))
            train_accuracy.append(train_epoch_accuracy)

            # Make predictions on the validation set
            valid_preds = model.predict(valid_indices)
            print("Probability label on validation data:", valid_preds)

            # Calculate validation accuracy
            # valid_epoch_accuracy = 100. * (accuracy_score(valid_labels, valid_preds))
            valid_epoch_accuracy = 100. * (model.score(valid_indices, valid_labels))
            valid_accuracy.append(valid_epoch_accuracy)

            print(f"Training acc: {train_epoch_accuracy:.3f}")
            print(f"Validation acc: {valid_epoch_accuracy:.3f}")
            print('-' * 50)

            # Calculate log loss for training and validation sets
            train_epoch_loss = 100. * (log_loss(train_labels, train_preds))
            train_loss.append(train_epoch_loss)
            valid_epoch_loss = log_loss(valid_labels, valid_preds)
            valid_loss.append(valid_epoch_loss)

            print(f"Training Loss: {train_epoch_loss:.3f}")
            print(f"Validation Loss: {valid_epoch_loss:.3f}")
            print('-' * 50)

        if epoch == epochs - 1:
            valid_preds_list.append(valid_preds.squeeze())
            print(f"Fold {fold + 1} - valid_preds_list:", valid_preds_list)
            valid_labels_list.append(valid_labels.cpu().numpy().squeeze())
            print(f"Fold {fold + 1} - valid_labels_list:", valid_labels_list)


        # Calculate AUC score
        print("valid_labels_list:", valid_labels_list[0])
        print("shape of valid_labels_list", np.shape(valid_labels_list[0]))
        print("valid_preds_list:", valid_preds_list[0])
        print("shape of valid_preds_list", np.shape(valid_preds_list[0]))
        auc_score = roc_auc_score(valid_labels_list[0], valid_preds_list[0])
        print("AUC_Score:", auc_score)

        # Calculate Balanced Accuracy Score
        balanced_acc = balanced_accuracy_score(valid_labels_list[0], valid_preds_list[0])
        print("Balance_Score:", balanced_acc)

        # Append the AUC score and Balanced Accuracy score to the fold_results list
        fold_results.append((auc_score, balanced_acc))

        print(f"Fold {fold + 1} - AUC Score: {auc_score:.4f}, Balanced Accuracy: {balanced_acc:.4f}")

        # Track the accuracies for each epoch.
        last_epoch_train_acc = train_accuracy[-1]
        print("Print last epoch training accuracy", last_epoch_train_acc)
        last_epoch_valid_acc = valid_accuracy[-1]
        print("Print last epoch validation accuracy", last_epoch_valid_acc)

        # Store the last epoch's training and validation accuracies for this fold.
        last_epoch_train_acc_list.append(last_epoch_train_acc)
        last_epoch_valid_acc_list.append(last_epoch_valid_acc)

        # Save the loss and accuracy plots.
        save_plots(
            train_accuracy,
            valid_accuracy,
            train_loss,
            valid_loss,
            name=plot_name
        )

        # Save the last epoch's training and validation accuracy to a CSV file.
        with open(f'/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/accuracy_fold_{fold}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Training Accuracy', 'Validation Accuracy'])
            for epoch, train_acc_epoch, valid_acc_epoch in zip(range(1, epochs + 1), train_accuracy, valid_accuracy):
                writer.writerow([epoch, train_acc_epoch, valid_acc_epoch])


        with open(f'/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/fold_{fold}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fold', 'AUC Score', 'Balanced Accuracy'])
            for i, (auc, bal_acc) in enumerate(fold_results):
                writer.writerow([i, auc, bal_acc])

    print('TRAINING COMPLETE')


def plot_result(x_label, y_label, plot_title, train_data, val_data):
    # Set size of plot
    plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold"]
    # labels = ["1st Fold", "2nd Fold"]
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
            last_epoch_train_acc_list,
            last_epoch_valid_acc_list)


# Make predictions on the test dataset.
test_preds_prob, test_preds_class = predict_test_data(model, test_loader, device)

# for i in preds:
for i in np.nditer(test_preds_class):
        print("Class probability for test class:", i)
        classification_results.append(i)  # Store the classification in the list

# Write the classification results to the CSV file
with open('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_PyTorch/Texture_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Classification'])

    # Write the classification results
    for result in classification_results:
        writer.writerow([result])

print("CSV file has been created successfully.")
