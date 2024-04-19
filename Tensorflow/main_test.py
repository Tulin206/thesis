# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import scipy
from scipy.io import loadmat
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
# Import ResNet 18 model
#import resnet18
# import ResNet18_modified
# Import ResNet 50 model
# import resnet50
import MobileNet
#import CNN_2D
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


# Creating Image Size of Spectral Data
IMAGE_SIZE = [128, 384, 441]
# 441 is the wavelength of spectra
# 128 is the height of the image
# 384 is the width of the image


# import Training_Data
import Training_Dataset
# import Training_Dataset_PCA
# Load the 10 training spectral datasets into a list
datasets = [np.load('trainingset_1.npy'), np.load('trainingset_2.npy'), np.load('trainingset_3.npy'),
            np.load('trainingset_4.npy'), np.load('trainingset_5.npy'), np.load('trainingset_6.npy'),
            np.load('trainingset_7.npy'), np.load('trainingset_9.npy'), np.load('trainingset_10.npy'),
            np.load('trainingset_8.npy'), np.load('trainingset_11.npy'), np.load('trainingset_12.npy'),
            np.load('trainingset_13.npy'), np.load('trainingset_16.npy'), np.load('trainingset_14.npy'), np.load('trainingset_15.npy')]

print('Shape of whole training datasets', np.shape(datasets))
print('\n')
print('End of printing the training datasets\n')


# import Test_Data
import Test_Dataset
# import Test_Dataset_PCA
# Load the 18 test spectral datasets into a list
testdatasets = [np.load('testdataset_1.npy'), np.load('testdataset_2.npy'), np.load('testdataset_3.npy'),
                np.load('testdataset_4.npy'), np.load('testdataset_5.npy'), np.load('testdataset_6.npy'),
                np.load('testdataset_7.npy'), np.load('testdataset_8.npy'), np.load('testdataset_9.npy'),
                np.load('testdataset_10.npy'), np.load('testdataset_11.npy'), np.load('testdataset_12.npy')]

print('Shape of whole test datasets', np.shape(testdatasets))
print('\n')
print('End of printing the test datasets\n')


x = np.array(datasets)
y = np.array(testdatasets)
#data = np.vstack([dataset1, dataset2, dataset3, dataset4, dataset5])

# Generate labels for each image
classes = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1])
print ('labels without one hot encoding\n', classes) # 0 represents soft class, 1 represents hard class
# classes = tf.one_hot(classes, depth=2)
# print('Labels with one hot encoding\n', classes)

# Set the labels of Training dataset
# X_train, y_train = x[0:14], classes[0:14]
Train_data, Label_Data = x[0:16], classes[0:16]

#Dimension of the training Spectral dataset
print('Shape of training dataset and label:')
print((np.shape(Train_data), Label_Data.shape))
print('\n')


# Compile ResNet18 model
MobileNet.model.compile(# optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        optimizer=tf.keras.optimizers.SGD(learning_rate = 0.0001, momentum = 0.9, nesterov = False),
                        loss='binary_crossentropy', metrics=[[tf.keras.metrics.AUC()], 'accuracy'])

# Compile ResNet50 model
# resnet50.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 0.0001, momentum = 0.9, nesterov = False), loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# Define the checkpoint filepath
# checkpoint_filepath = 'model_weights_epoch_{epoch:02d}.h5'
# checkpoint_filepath = 'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'

# checkpoint
checkpoint_filepath = "weights.best.hdf5"

# Define the ModelCheckpoint callback to save weights after each epoch
checkpoint_callback = ModelCheckpoint(
    checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss', # 'val_auc',  # validation AUC # 'val_accuracy',
    mode='min',
    verbose=1
)

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
Skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# print('SKfold status: ', Skfold)
print('\n')


# Define the K-fold cross-validation strategy
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
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
    val_aug = train_generator.flow(X_val, y_val)
    # print(np.shape(val_aug))


    # Create a new CSVLogger for this fold
    log_csv = CSVLogger(f'fold_{i}.csv', separator=',', append=False)


    # Training the model with ResNet 18
    history = MobileNet.model.fit(train_aug,
                                 # X_train, y_train,
                                 batch_size=1,
                                 # steps_per_epoch=10,
                                 epochs=100,
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
    train_accuracy = MobileNet.model.evaluate(train_aug, verbose=1)
    print('/n/n')
    print('Train Loss, AUC and Training Accuracy', train_accuracy)
    val_accuracy = MobileNet.model.evaluate(X_val, y_val, verbose=1)
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


    # Predict the probabilities of the Validation data
    y_pred_proba = MobileNet.model.predict(X_val)


    # calculate AUC score for each fold
    from sklearn.metrics import roc_auc_score
    try:
        auc_score = roc_auc_score(y_val, y_pred_proba)
    except ValueError:
        pass

    auc_score = roc_auc_score(y_val, y_pred_proba)
    print('roc_aoc_score', auc_score)
    Auc_Score.append(auc_score)

    # Calculate Balanced Accuracy score for each fold
    from sklearn.metrics import balanced_accuracy_score
    balance_score = balanced_accuracy_score(y_val, np.round(abs(y_pred_proba)))
    print('Balanced_Accuracy_Score', balance_score)
    Balance_Score.append(balance_score)
    print('\n \n')

# Create a DataFrame to store the accuracy for each fold:
for fold, (train_index, val_index) in enumerate(Skfold.split(Train_data, Label_Data)):
    # Store scores for this fold
    fold_scores.append((fold + 1, Balance_Score, Auc_Score))


# Convert fold_scores to a DataFrame
df_scores = pd.DataFrame(fold_scores, columns=['Fold', 'Balance Score', 'Accuracy Score'])

# Save scores to a CSV file
df_scores.to_csv('cross_validation_scores.csv', index=False)

import csv

with open('metrics.csv', mode='w') as csv_file:
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
    plt.savefig(f'Loss_acc_{fold}.png')
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
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
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
    plt.savefig('accuracy_per_fold_plot.png')


# Plot Accuracy Result
model_name = "Decision Tree"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            # mean_train_acc,
            # mean_val_acc)
            last_epoch_accuracy,
            last_epoch_val_accuracy)


# Prediction on Test Datasets using ResNet18
preds = MobileNet.model.predict(y)
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
with open('Texture_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Classification'])

    # Write the classification results
    for result in classification_results:
        writer.writerow([result])

print("CSV file has been created successfully.")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/