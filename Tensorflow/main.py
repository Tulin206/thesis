# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
import scipy
from scipy.io import loadmat
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
# Import ResNet 18 model
import resnet18
# import ResNet18_modified
# Import ResNet 50 model
# import resnet50
import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
import keras
from keras import datasets, layers, models
from keras import initializers
from keras.models import Sequential, load_model, Model
from keras.utils import CustomObjectScope
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.utils import array_to_img, img_to_array, load_img
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from IPython.display import SVG
# from tensorflow.keras.utils.vis_utils import model_to_dot
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from glob import glob
from statistics import mean

# Configures TensorFlow ops to run deterministically.
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#
# tf.debugging.set_log_device_placement(True)

# # Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
#
# print(c)


# Creating Image Size of Spectral Data
IMAGE_SIZE = [128, 384, 441]
# 441 is the wavelength of spectra
# 128 is the height of the image
# 384 is the width of the image


# import Training_Data
import Training_Dataset
# import Training_Dataset_PCA


# import Test_Data
import Test_Dataset
# import Test_Dataset_PCA


x = np.array(Training_Dataset.datasets)
y = np.array(Test_Dataset.testdatasets)
#data = np.vstack([dataset1, dataset2, dataset3, dataset4, dataset5])

# Generate labels for each image
classes = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
# classes = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
print ('labels without one hot encoding\n', classes) # 0 represents soft class, 1 represents hard class
# classes = tf.one_hot(classes, depth=2)
# print('Labels with one hot encoding\n', classes)

# Set the labels of Training dataset
# X_train, y_train = x[0:14], classes[0:14]
# Train_data, Label_Data = x[0:11], classes[0:11]
Train_data, Label_Data = x[0:16], classes[0:16]

#Dimension of the training Spectral dataset
print('Shape of training dataset and label:')
print((np.shape(Train_data), Label_Data.shape))
print('\n')


# Compile ResNet18 model
resnet18.model.compile(# optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       optimizer=tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9, nesterov = False),
                        #optimizer='adam',
                        loss='binary_crossentropy', metrics=[[tf.keras.metrics.AUC()], 'accuracy'])

# Compile ResNet50 model
# resnet50.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 0.0001, momentum = 0.9, nesterov = False), loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# Define the checkpoint filepath
# checkpoint_filepath = 'model_weights_epoch_{epoch:02d}.h5'
# checkpoint_filepath = 'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'

# checkpoint
checkpoint_filepath = "/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_CSV/weights.best.hdf5"

# Define the ModelCheckpoint callback to save weights after each epoch
checkpoint_callback = ModelCheckpoint(
    checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss', # 'val_auc',  # validation AUC # 'val_accuracy',
    mode='min',
    verbose=1
)



#Learning Rate Annealer
from keras.callbacks import ReduceLROnPlateau
lrr= ReduceLROnPlateau(monitor='val_accuracy',   factor=.01,   patience=3,  min_lr=0.01)



early_stop = EarlyStopping(monitor='val_loss', #'val_auc', #'val_accuracy',
                           patience=5, mode='min', verbose=1)

# CSVLogger logs epoch, acc, loss, val_acc, val_loss
# log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

# Image Data Augmentation
train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, shear_range=0.2, zoom_range=0.2)

# Load the data and split it into training and validation sets
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# #Train-validation-test split
# X_val, y_val = x[14:], classes[14:]

#Dimension of the validation Spectral dataset
    # print('Shape of validation dataset and label:')
    # print((np.shape(X_val), y_val.shape))
    # print('\n')



# Define the stratified K-fold cross-validation strategy
from sklearn.model_selection import cross_val_score, StratifiedKFold
Skfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
# print('SKfold status: ', Skfold)
print('\n')


# Define the K-fold cross-validation strategy
from sklearn.model_selection import KFold
kfold = KFold(n_splits=4, shuffle=True, random_state=42)
# kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

# Store the list of training and validation accuracy and training and validation loss
train_acc = []
val_acc = []
train_loss = []
val_loss = []
# mean_train_acc = []
# mean_val_acc = []
training_accuracy = []
validation_accuracy = []
results = []
last_epoch_accuracy = []
last_epoch_val_accuracy = []
last_epoch_loss = []
last_epoch_val_loss = []
fold_scores = []
classification_results = []  # List to store the classification results
Auc_Score = []
Balance_Score = []

# Training dataset for each folder (Kfold = 5)
for i, (train_index, val_index) in enumerate(Skfold.split(Train_data, Label_Data)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Val:  index={val_index}")
    X_train, X_val, y_train, y_val = Train_data[train_index], Train_data[val_index], Label_Data[train_index], Label_Data[val_index]

    print('\n')
    print('Shape of split training dataset and label:')
    print((np.shape(X_train), y_train.shape))

    print('Shape of split validation dataset and label:')
    print((np.shape(X_val), y_val.shape))
    print('\n')

    # Fitting the augmentation defined above to the data
    train_aug = train_generator.flow(X_train, y_train, batch_size=1)
    # print(np.shape(train_aug))
    # val_aug = train_generator.flow(X_val, y_val)
    # print(np.shape(val_aug))


    # Create a new CSVLogger for this fold
    log_csv = CSVLogger(f'/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_CSV/fold_{i}.csv', separator=',', append=False)


    # Training the model with ResNet 18
    history = resnet18.model.fit(train_aug,
                                 # X_train, y_train,
                                 batch_size=1,
                                 # steps_per_epoch=10,
                                 epochs=1000,
                                 validation_data=(X_val, y_val),
                                 # validation_steps=5,
                                 callbacks=[checkpoint_callback, log_csv])


    # Training the model with ResNet 50
    # history = resnet50.model.fit(train_aug,
    # batch_size=1,
    # epochs = 100,
    # validation_data=(X_val, y_val),
    # callbacks=[checkpoint_callback, early_stop, log_csv])

    # Check the training and validation accuracy
    train_accuracy = resnet18.model.evaluate(train_aug, verbose=1)
    print('/n/n')
    print('Train Loss, AUC and Training Accuracy', train_accuracy)
    val_accuracy = resnet18.model.evaluate(X_val, y_val, verbose=1)
    print('Validation Loss, AUC and Validation Accuracy', val_accuracy)

    # Appending training accuracy and validation accuracy into separate list to plot those data for each fold
    training_accuracy.append(train_accuracy[2])
    validation_accuracy.append(val_accuracy[2])

    # Appending the value of training and validation accuracy into a list to plot these data for each epoch
    train_acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])

    # Appending the value of training and validation loss into a list
    train_loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])

    last_epoch_metrics = history.history
    last_epoch_accuracy.append(last_epoch_metrics['accuracy'][-1])
    last_epoch_val_accuracy.append(last_epoch_metrics['val_accuracy'][-1])
    last_epoch_loss.append(last_epoch_metrics['loss'][-1])
    last_epoch_val_loss.append(last_epoch_metrics['val_loss'][-1])

    # Append the results to the results list
    results.append([i, train_acc, val_acc, train_loss, val_loss])

    # # Calculate the mean training and validation accuracy across all folds
    # mean_train_acc.append(np.mean(train_acc))
    # print('mean of Training Accuracy:', mean_train_acc)
    # mean_val_acc.append(np.mean(val_acc))
    # print('mean of Validation Accuracy:', mean_val_acc)

    # train_accuracy = resnet50.model.evaluate(train_aug, verbose=0)
    # print("Training Accuracy: {:.2f}%".format(train_accuracy*100))
    print('\n')
    print("Fold:", i, 'Training Accuracy', train_accuracy[2])
    print("Fold:", i, 'validation Accuracy', val_accuracy[2])
    print("Fold:", i, 'Training Loss', train_accuracy[0])
    print("Fold:", i, 'Validation Loss', val_accuracy[0])
    print('\n')



    # Define the file path and name
    # import csv
    # file_name = "losses.csv"
    #
    # # Open the file in write mode
    # with open(file_name, 'w', newline='') as csvfile:
    #     # Define the field names
    #     fieldnames = ['Epoch', 'Training Loss', 'Validation Loss']
    #
    #     # Create a writer object and write the header row
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    #     # Write the data row by row
    #     for j in range(len(train_loss)):
    #         writer.writerow({'Epoch': j + 1, 'Training Loss': train_loss[j], 'Validation Loss': val_loss[j]})
    #
    # file_name = "accuracy.csv"
    #
    # # Open the file in write mode
    # with open(file_name, 'w', newline='') as csvfile:
    #     # Define the field names
    #     fieldnames = ['Epoch', 'Training Accuracy', 'Validation Accuracy']
    #
    #     # Create a writer object and write the header row
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    #     # Write the data row by row
    #     for j in range(len(train_acc)):
    #         writer.writerow({'Epoch': j + 1, 'Training Accuracy': train_acc[j], 'Validation Accuracy': val_acc[j]})

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # t = f.suptitle('Deep Neural Net Performance', fontsize=12)
    # f.subplots_adjust(top=0.85, wspace=0.3)


# need to uncomment if necessary
    # epochs = list(range(1, 2 + 1))
    # ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    # ax1.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    # ax1.set_xticks(epochs)
    # ax1.set_ylabel('Accuracy Value')
    # ax1.set_xlabel('Epoch')
    # ax1.set_title('Accuracy')
    # l1 = ax1.legend(loc="best")
    #
    # ax2.plot(epochs, history.history['loss'], label='Train Loss')
    # ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    # ax2.set_xticks(epochs)
    # ax2.set_ylabel('Loss Value')
    # ax2.set_xlabel('Epoch')
    # ax2.set_title('Loss')
    # l2 = ax2.legend(loc="best")
    # plt.savefig(f'Loss_acc_{i}.png')
    # plt.show()
    #
    # # Clear the plot for the next fold
    # plt.clf()
    # plt.close()


    # # plot training and validation accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(['Training Accuracy', 'validation Accuaracy'], loc="best") #'upper left')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f'plot_acc_{i}.png')
    #
    # # plot training and validation loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(['Training Loss', 'Validation Loss'], loc="best")
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f'plot_loss_{i}.png')


    # # plotting accuracy and loss for each fold
    # print("Fold:", i, 'Accuracy and Loss curve')
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # t = f.suptitle('Deep Neural Net Performance', fontsize=12)
    # f.subplots_adjust(top=0.85, wspace=0.3)
    #
    # epochs = list(range(1, 100 + 1))
    # ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    # ax1.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    # ax1.set_xticks(epochs)
    # ax1.set_ylabel('Accuracy Value')
    # ax1.set_xlabel('Epoch')
    # ax1.set_title('Accuracy')
    # l1 = ax1.legend(loc="best")
    #
    # ax2.plot(epochs, history.history['loss'], label='Train Loss')
    # ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    # ax2.set_xticks(epochs)
    # ax2.set_ylabel('Loss Value')
    # ax2.set_xlabel('Epoch')
    # ax2.set_title('Loss')
    # l2 = ax2.legend(loc="best")

    # Predict the probabilities of the Validation data
    y_pred_proba = resnet18.model.predict(X_val)


    # calculate AUC score for each fold
    from sklearn.metrics import roc_auc_score
    try:
        auc_score = roc_auc_score(y_val, y_pred_proba)
        print('roc_aoc_score', auc_score)
        Auc_Score.append(auc_score)
    except ValueError:
        pass

    # auc_score = roc_auc_score(y_val, y_pred_proba)
    # print('roc_aoc_score', auc_score)

    # Calculate Balanced Accuracy score for each fold
    from sklearn.metrics import balanced_accuracy_score
    balance_score = balanced_accuracy_score(y_val, np.round(abs(y_pred_proba)))
    print('Balanced_Accuracy_Score', balance_score)
    Balance_Score.append(balance_score)
    print('\n \n')

    # Create a DataFrame to store the accuracy for each fold:
    fold_scores.append((i, Balance_Score, Auc_Score))

    # Convert fold_scores to a DataFrame
    df_scores = pd.DataFrame(fold_scores, columns=['Fold', 'Balance Score', 'Accuracy Score'])
    # Save scores to a CSV file
    df_scores.to_csv('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_CSV/cross_validation_scores.csv', index=False)

# # Create a DataFrame to store the accuracy for each fold:
# for fold, (train_index, val_index) in enumerate(Skfold.split(Train_data, Label_Data)):
#     # Store scores for this fold
#     fold_scores.append((fold + 1, Balance_Score, Auc_Score))

    # results_df = pd.DataFrame({'Fold': i,
    #                            'AUC Score': auc_score,
    #                            'Balance Score': balance_score},
    #                           index=[0])
    #
    # # Append the results to a CSV file
    # results_df.to_csv('kfold_results.csv', mode='a', header=(i == 1), index=False)
    #
    # i += 1

# # Convert fold_scores to a DataFrame
# df_scores = pd.DataFrame(fold_scores, columns=['Fold', 'Balance Score', 'Accuracy Score'])

# # Save scores to a CSV file
# df_scores.to_csv('cross_validation_scores.csv', index=False)

import csv

with open('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_CSV/metrics.csv', mode='w') as csv_file:
    fieldnames = ['metric', 'value']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for train_a, val_a, train_l, val_l in zip(last_epoch_accuracy, last_epoch_val_accuracy, last_epoch_loss, last_epoch_val_loss):
        writer.writerow({'metric': 'train_accuracy', 'value': train_a})
        writer.writerow({'metric': 'val_accuracy', 'value': val_a})
        writer.writerow({'metric': 'train_loss', 'value': train_l})
        writer.writerow({'metric': 'val_loss', 'value': val_l})

# Prediction on Test Datasets using ResNet50
# preds = resnet50.model.predict(y)
# preds =  np.argmax(preds, axis = 1)[0:12]
# print('labels of Test data with ArgMax:', preds)
# print('\n')

print('Training Accuracy for each epoch', train_acc)
# print('mean of Training Accuracy:', mean_train_acc)
print('\n \n')
print('Validation Accuracy for each epoch', val_acc)
# print('mean of Validation Accuracy:', mean_val_acc)

# # For appending data vertically instead of horizantally
# train_acc_formatted_list = '\n'.join([str(element) for element in train_acc])
# val_acc_formatted_list = '\n'.join([str(element) for element in val_acc])
# train_loss_formatted_list = '\n'.join([str(element) for element in train_loss])
# val_loss_formatted_list = '\n'.join([str(element) for element in val_loss])

# # create plot for each fold
# import matplotlib.pyplot as plt
#
# fig, axs = plt.subplots(5, sharex=True, sharey=True)
# fig.suptitle('Training and Validation Accuracy for each fold and epoch')
#
# for i in range(5):  # assuming 5-fold cross-validation
#     axs[i].plot(train_acc_values[i], label='Training')
#     axs[i].plot(val_acc_values[i], label='Validation')
#     axs[i].set_title(f'Fold {i+1}')
#     axs[i].legend()
#
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()


# create plot for each fold
for fold, (train_index, val_index) in enumerate(Skfold.split(Train_data, Label_Data)):
    # get accuracy values for this fold
    train_fold = train_acc[fold]
    val_fold = val_acc[fold]
    train_loss_fold = train_loss[fold]
    val_loss_fold = val_loss[fold]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    t = f.suptitle('Deep Neural Net Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    # create plot
    epochs = np.arange(len(train_fold)) + 1
    ax1.plot(epochs, train_fold, label='Training Accuracy')
    ax1.plot(epochs, val_fold, label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Fold {} Accuracy_Curve'.format(fold+1))
    l1 = ax1.legend(loc="best")

    ax2.plot(epochs, train_loss_fold, label='Training Loss')
    ax2.plot(epochs, val_loss_fold, label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Fold {} Loss_Curve'.format(fold+1))
    l2 = ax2.legend(loc="best")
    plt.savefig(f'/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_CSV/Loss_acc_{fold}.png')
    # plt.show()

# # Define the file path and name
# import csv
#
# # write accuracy lists to CSV file
# with open('accuracy.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Training Accuracy', 'Validation Accuracy'])
#     for train_acc, val_acc in zip(train_acc_formatted_list, val_acc_formatted_list):
#         writer.writerow([train_acc_formatted_list, val_acc_formatted_list])
#
# # write accuracy lists to CSV file
# with open('losses.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Training Loss', 'Validation Loss'])
#     for train_acc, val_acc in zip(train_loss_formatted_list, val_loss_formatted_list):
#         writer.writerow([train_loss_formatted_list, val_loss_formatted_list])

# # After the k-fold loop is finished, create a Pandas DataFrame from the results list:
# import pandas as pd
# results_df = pd.DataFrame(results, columns=['fold', 'train_acc', 'val_acc', 'train_loss', 'val_loss'])
#
# # Save the DataFrame to a CSV file:
# results_df.to_csv('results.csv', index=False)

# # Create Box_Plot for both training and validation accuracy
# fig, ax = plt.subplots()
# ax.boxplot([mean_train_acc, mean_val_acc])
# ax.set_xticklabels(['Train', 'Validation'])
# ax.set_ylabel('Accuracy')
# ax.set_title('5-Fold Cross-Validation Accuracy')
#
# # Show plot
# plt.show()


# Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data):
    '''Function to plot a grouped bar chart showing the training and validation
      results of the ML model in each fold after applying K-fold cross-validation.
     Parameters
     ----------
     x_label: str,
        Name of the algorithm used for training e.g 'Decision Tree'

     y_label: str,
        Name of metric being visualized e.g 'Accuracy'
     plot_title: str,
        This is the title of the plot e.g 'Accuracy Plot'

     train_result: list, array
        This is the list containing either training precision, accuracy, or f1 score.

     val_result: list, array
        This is the list containing either validation precision, accuracy, or f1 score.
     Returns
     -------
     The function returns a Grouped Barchart showing the training and validation result
     in each fold.
    '''

    # Set size of plot
    plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold"] #"5th Fold"]
    X_axis = np.arange(len(labels))
    ax = plt.gca()
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_CSV/accuracy_per_fold_plot.png')


# Plot Accuracy Result
model_name = "Bar Chart"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            # mean_train_acc,
            # mean_val_acc)
            last_epoch_accuracy,
            last_epoch_val_accuracy)


# Prediction on Test Datasets using ResNet18
preds = resnet18.model.predict(y)
print('labels of test data without argmax\n', preds)
print(type(preds))
print('\n')


# for i in preds:
for i in np.nditer(preds):
        print(i)
        if i < 0.5:
            # print("The image classified is soft texture")
            classification = "The image is classified as soft texture"
            # writer.writerow([message])
        else:
            # print("The image classified is hard texture")
            classification = "The image is classified as hard texture"
            # writer.writerow([message])

        classification_results.append(classification)  # Store the classification in the list

# Write the classification results to the CSV file
with open('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/Result_Plot_CSV/Texture_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Classification'])

    # Write the classification results
    for result in classification_results:
        writer.writerow([result])

print("CSV file has been created successfully.")


# # Predict the probabilities of the training data
# y_pred_proba = resnet18.model.predict(Train_data)
#
#
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve, auc
#
#
# print('roc_aoc_score', roc_auc_score(Label_Data, y_pred_proba))
#
# # Compute the ROC curve and AUC using the roc_curve and auc functions from scikit-learn.
# fpr, tpr, thresholds = roc_curve(Label_Data, y_pred_proba)
# roc_auc = auc(fpr, tpr)
# print('roc_auc', roc_auc)
#
#
# # Plot the ROC curve using the matplotlib library.
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# # plt.show()
# plt.savefig('ROC_AUC_plot.png')
#
#
# from sklearn.metrics import balanced_accuracy_score
# print('Balanced_Accuracy_Score', balanced_accuracy_score(Label_Data, np.round(abs(y_pred_proba))))
# print('\n \n')


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/