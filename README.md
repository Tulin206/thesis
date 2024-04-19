## Deep-learning based quantification of pancreatic texture using IR-Spectroscopy data

## Project Description: 
This project implements various deep learning techniques using PyTorch to classify pancreas texture and pancreatic tumor grade using IR Spectroscopy data. Both binary and multi-class classification for pancreas texture are implemented, with some experiments conducted using the TensorFlow framework. All experiment are implemented with the already established deep neural network without making any fine-tune changes. 

## Repository: 
This repository contains source code for classifying both pancreas texture and pnacreatic tumor grade using both conventional machine learning technique and deep learning approaches. Each folder has sub-folders, where each experiment is conducted and script starting with "**main_***" executes the entire project. Upon execution, training/validation loss curves, accuracy plots, CSV files of evaluation metric, and saliency maps, showing regions of focus for class prediction, are stored in directories.

## Installation
pip install -r requirements.txt

## Evaluation
For both pancreas texture and tumor grade classification using binary classifiers, the ResNet 18 model outperformed ResNet 18 with pre-trained weights, MobileNet V1, and Logistic Regression models.

## Performance matrics for binary classification of pancreas texture using stratified 6-fold cross validation for initially 16 samples:
	ResNet 18
		- Mean Validation Accuracy: 88%
		- Mean AUC Score: 92%
		- Mean Balanced Accuracy: 83%
	
	K-Means + ResNet 18
		- Mean Validation Accuracy: 74%
		- Mean AUC Score: 88%
		- Mean Balanced Accuracy: 63%

	MobileNet V1
		- Mean Validation Accuracy: 51%
		- Mean AUC Score: 75%
		- Mean Balanced Accuracy: 67%

	Logistic Regression
		- Mean Validation Accuracy: 66%
		- Mean AUC Score: 58%
		- Mean Balanced Accuracy: 58%

	ResNet 18 with Pre-trained weights
		- Mean Validation Accuracy: 33%
		- Mean AUC Score: 25%
		- Mean Balanced Accuracy: 25%

## Performance matrics for binary classification of panreatic tumor grade using stratified 8-fold cross validaton of 27 samples:
	ResNet 18
		- Mean Validation Accuracy: 88%
		- Mean AUC Score: 97%
		- Mean Balanced Accuracy: 88%
