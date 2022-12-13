import numpy as np
from skimage.transform import resize
from Data_Utils import load_nii
from Config.Data_Config import ori_data_path,ori_label_path
import os
import matplotlib.pyplot as plt
from batchgenerators.augmentations.utils import resize_segmentation
data_path = os.path.join(ori_data_path,'CaoYanYan','CaoYanYan.nii.gz')
label_path = os.path.join(ori_label_path,'CaoYanYan','CaoYanYan_gt.nii.gz')

img,_,__ = load_nii(data_path)
label,_,__ = load_nii(label_path)
# resized_label = resize_segmentation(label,(128,128,128))
# print(label.shape,resized_label.shape)
# print(np.unique(resized_label))
# print(np.sum(label==1))
# print(np.sum(resized_label==1))
# plt.figure()
# # for i in range(16):
# #     plt.imshow(resized_label[:,:,i],cmap='gray')
# #     plt.show()
# for i in range(120,200):
#     plt.subplot(1,2,1)
#     plt.imshow(label[:,:,i],cmap='gray')
#     plt.subplot(1,2,2)
#     plt.imshow(resized_label[:, :, i//2], cmap='gray')
#     plt.show()