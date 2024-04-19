import sys

# To train the model with initial 16 samples using 6-fold cross-validation
sys.path.append('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/pycharm_proj_temp/PythonProject_Test/Pytorch/Binary_MobileNet_PyTorch/sample_16/')

import sample_16.S_6_fold_cv

if __name__ == '__main__':
    sample_16.S_6_fold_cv.S_6_fold_cv()
    sample_16.S_6_fold_cv.process_result()


# # To train the model with the entire 28 samples using 8-fold cross-validation
# sys.path.append('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/pycharm_proj_temp/PythonProject_Test/Pytorch/Binary_MobileNet_PyTorch/sample_28/')
#
# import sample_28.S_8_fold_cv
#
# if __name__ == '__main__':
#     sample_28.S_8_fold_cv.S_8_fold()
#     sample_28.S_8_fold_cv.process_result()
