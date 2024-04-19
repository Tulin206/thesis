The ResNet 18 model undergoes training with both the dataset containing outlier spectra and the dataset after removing them, facilitating a comparison of the model's performance in both scenarios.

Initiallay 16 samples are utilized to trian the model. Later after adding 12 more samples, the deep neural network model is trained with the entire 28 samples of pancreas tissue to classify texture. 

A combination of K-Means clustering with ResNet 18 is also employed to remove 5 low-quality samples from 16 pancreas texture samples. __DataLoad_PyTorch.py__ file from __sample_28__ folder contains the information of indices of low quality samples with corresponding labels.

From __main_new_PyTorch.py__, all the results of stratified cross-validation and leave-One-Out cross-validation with both 16 samples and 28 samples can be produced. To train the model with 16 samples, `import sample_16.stratified_k_fold_cv` should be executed. For training the model with 28 samples using 8-fold cross-validation `import sample_28.S_8_fold_cv` should be executed and for leave-one-out cross-validation, `import sample_28.loo_cv` should be executed.
