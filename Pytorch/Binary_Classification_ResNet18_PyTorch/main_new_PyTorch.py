import sys

# # To train the model with initial 16 samples using 6-fold cross-validation
# sys.path.append('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/pycharm_proj_temp/PythonProject_Test/Pytorch/Binary_Classification_ResNet18_PyTorch/sample_16/')
#
# import sample_16.stratified_k_fold_cv
#
# if __name__ == '__main__':
#     sample_16.stratified_k_fold_cv.sk_cv()
#     sample_16.stratified_k_fold_cv.process_result()


# To train the model with the entire 28 samples using 8-fold cross-validation
sys.path.append('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/pycharm_proj_temp/PythonProject_Test/Pytorch/Binary_Classification_ResNet18_PyTorch/sample_28/')
import sample_28.S_8_fold_cv

if __name__ == '__main__':
    sample_28.S_8_fold_cv.s_8_fold()
    sample_28.S_8_fold_cv.process_result()


# # To train the model with the entire 28 samples using Leave-One-Out cross-validation
# sys.path.append('/mnt/ceph/tco/TCO-Students/Homes/ISRAT/pycharm_proj_temp/PythonProject_Test/Pytorch/Binary_Classification_ResNet18_PyTorch/sample_28/')
# import sample_28.loo_cv
#
# if __name__ == '__main__':
#     sample_28.loo_cv.loo_cv()
#     sample_28.loo_cv.process_result()

