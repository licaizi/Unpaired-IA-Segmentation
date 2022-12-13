import os
import nibabel as nib
import itk
import numpy as np
from itkwidgets import view
def load_data(path):
    nimage = nib.load(path)
    return nimage.get_data(),nimage.affine,nimage.header

# data_path = '../../../../Datasets/'
# ct_path = data_path + 'ct_train1/'
DSA_PATH = '/home/hci/Datasets/Aneurysm/DSA/crop_data/'
# print(os.listdir(data_path))
data_paths = os.listdir(DSA_PATH)
print(np.random.uniform(0.1,0.3))
print(data_paths.sss)
for data_path in data_paths:
    src_name = DSA_PATH + data_path+'/'+data_path+'-label.nii.gz'
    dst_name = DSA_PATH + data_path+'/'+data_path+'_gt.nii.gz'
    # '{}_gt.nii.gz'.format(path)
    # os.rename(src_name,dst_name)
    # image = itk.imread(DSA_PATH + data_path+'/'+data_path+'.nii.gz')

    data,affine,header = load_data(DSA_PATH + data_path+'/'+data_path+'.nii.gz')
    gt, affine, header = load_data(DSA_PATH + data_path + '/' + data_path + '_gt.nii.gz')
    # view(data)
    # print('name:', data_path, gt.shape)
    # print(gt.shape)