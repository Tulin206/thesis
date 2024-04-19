import sys

# # To train the model with initial 16 samples using 6-fold cross-validation
# sys.path.append('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/pycharm_proj_temp/PythonProject_Test/Pytorch/Transfer_Learning_ResNet18_PyTorch/sample_16/')
#
# import sample_16.samples_16_cv_transferlearning
#
# if __name__ == '__main__':
#     sample_16.samples_16_cv_transferlearning.samples_16_cv_transferlearning()
#     sample_16.samples_16_cv_transferlearning.process_result()


# To train the model with initial 16 samples using 6-fold cross-validation
sys.path.append('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/pycharm_proj_temp/PythonProject_Test/Pytorch/Transfer_Learning_ResNet18_PyTorch/sample_28/')

import sample_28.samples_28_loo_transferlearning

if __name__ == '__main__':
    sample_28.samples_28_loo_transferlearning.samples_28_loo_transferlearning()
    sample_28.samples_28_loo_transferlearning.process_result()
