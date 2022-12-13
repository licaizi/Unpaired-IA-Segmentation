import os
import random
import time
import torch
from data_preprocessing.Data_Utils import center_crop_3D_image_batched,save_nii,center_crop_3D_image_gt,roi_crop_data
import numpy as np
from time import ctime
import threading
import queue
from queue import Queue
import json
from config.Data_Config import ori_data_path, ori_label_path, train_list_path, \
    val_list_path,dsa_label_path,dsa_data_path,val_dsa_list_path,train_dsa_list_path # , xls_path, img_name, gt_name
from data_preprocessing.Data_Utils import convert_to_one_hot, normalize_img, load_nii, get_orientation, \
    windwo_transform, normalize_img_after_windowtransform

deleted_datalist = ['HongXiLin', 'JinQiuXiang', 'WangYouMu', 'XuWeiRong', 'ZongGuiFang']
new_deleted_datalist = ['CaiGaoMing','ChenMeiLan3','ChenYueE','FuYaLin','YaoXiaoHong']
filtered_CTA_datalist = ['ChenMeiLan2','ChenMeiLan3','ChenYiYan','HongXiLin','JinQiuXiang','PengChengJiong','ChenYinDuan','YaoXiaoHong'
                     ,'ZhangSanBao','ChenYueE','WangQuanYi','RaoZhuXiao','WuTao','ZhangTingMei']
filtered_DSA_datalist = ['lixueqing-5','luojiawen-5','zhangchuhui','zhangjunyin2','liujinyu']

def generate_data_list():
    random.seed(2333)
    train_txt = open(train_dsa_list_path,'w')
    test_txt = open(val_dsa_list_path, 'w')
    data_list = os.listdir(dsa_data_path)
    random.shuffle(data_list)
    train_list = data_list[:44]
    train_names = ''
    for name in train_list:
        train_names += str(name)+'\n'
    train_txt.write(train_names)
    test_list = data_list[44:]
    test_names = ''
    for name in test_list:
        test_names += str(name)+'\n'
    test_txt.write(test_names)

# generate_data_list()
class Mythread(threading.Thread):
    def __init__(self,func,args=()):
        super(Mythread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):

        try:
            return self.result
        except Exception:
            return None

q = queue.Queue()
def read_data(path,data_type='CTA'):
    img_name = '{}.nii.gz'.format(path)
    gt_name = '{}_gt.nii.gz'.format(path)
    if data_type == 'DSA':
        img_path = os.path.join(dsa_data_path, path, img_name)
        gt_path = os.path.join(dsa_label_path, path, gt_name)
    else:
        img_path = os.path.join(ori_data_path, path, img_name)
        gt_path = os.path.join(ori_label_path, path, gt_name)
    img_data = load_nii(img_path)
    gt_data = load_nii(gt_path)
    return path,img_data,gt_data


def mutithread_get_data(norm=True, one_hot=False, window_width=700, window_center=80,Data_type = 'CTA',q_len=20,pt_center=(128,128,128)):
    '''
    param q_len:number of threads
    '''
    print('time:..........',ctime())
    if Data_type == 'DSA':
        paths = os.listdir(dsa_data_path)
        # print(paths)
        with open(train_dsa_list_path, 'r') as f:
            train_list = f.readlines()
            train_list = [x.rstrip('\n') for x in train_list]
        with open(val_dsa_list_path, 'r') as f:
            val_list = f.readlines()
            val_list = [x.rstrip('\n') for x in val_list]
    else:
        paths = os.listdir(ori_data_path)
        with open(train_list_path, 'r') as f:
            train_list = f.readlines()
            train_list = [x.rstrip('\n') for x in train_list]
        with open(val_list_path, 'r') as f:
            val_list = f.readlines()
            val_list = [x.rstrip('\n') for x in val_list]
    paths.sort()
    imgs = []
    dataset_train = {}
    dataset_val = {}
    for i in range(len(paths)):
        path = paths[i]
        # print('path:',path)
        t = Mythread(read_data,(path,Data_type))
        q.put(t)

        if q.qsize() == q_len or i == len(paths) -1:
            join_threads = []
            while not q.empty():
                thread = q.get()
                join_threads.append(thread)
                thread.start()
            #kill the threads
            for t in join_threads:
                t.join()
                path, img_data, gt_data = t.get_result()
                if path in train_list:
                    dataset_train[path] = {}
                    dataset_train[path]['center'] = [[pt_center[0], pt_center[1], pt_center[2]]]
                    img = img_data[0]
                    if Data_type == 'CTA':
                        img = windwo_transform(img, window_width, window_center)
                    img = np.transpose(img, (2, 0, 1))
                    gt = np.transpose(gt_data[0], (2, 0, 1))
                    if Data_type == 'DSA':
                        dataset_train[path]['img'] = normalize_img(img) if norm else img
                    elif Data_type == 'CTA':
                        dataset_train[path]['img'] = normalize_img_after_windowtransform(img, window_center,
                                                                                         window_width) if norm else img
                    dataset_train[path]['gt'] = convert_to_one_hot(gt) if one_hot else np.expand_dims(gt,0)
                    dataset_train[path]['nii_'] = [gt_data[1], gt_data[2]]
                elif path in val_list:
                    dataset_val[path] = {}
                    dataset_val[path]['center'] = [[pt_center[0], pt_center[1], pt_center[2]]]
                    img = img_data[0]
                    if Data_type == 'CTA':
                        img = windwo_transform(img, window_width, window_center)
                    img = np.transpose(img, (2, 0, 1))
                    gt = np.transpose(gt_data[0], (2, 0, 1))
                    if Data_type == 'DSA':
                        dataset_val[path]['img'] = normalize_img(img) if norm else img
                    elif Data_type == 'CTA':
                        dataset_val[path]['img'] = normalize_img_after_windowtransform(img, window_center,
                                                                                       window_width) if norm else img
                    dataset_val[path]['gt'] = convert_to_one_hot(gt) if one_hot else np.expand_dims(gt,0)
                    dataset_val[path]['nii_'] = [gt_data[1], gt_data[2]]

    print('after collecting data:',ctime())
    # print('lens:',dataset_train.keys(),dataset_val.keys())
    return dataset_train, dataset_val

def get_labeled_data(norm=True, one_hot=False, window_width=700, window_center=80,Data_type = 'CTA',pt_center=(128,128,128),norm_type='z-score'):
    print(ctime())
    if Data_type == 'DSA':
        paths = os.listdir(dsa_data_path)
        # print(paths)
        with open(train_dsa_list_path, 'r') as f:
            train_list = f.readlines()
            train_list = [x.rstrip('\n') for x in train_list]
        with open(val_dsa_list_path, 'r') as f:
            val_list = f.readlines()
            val_list = [x.rstrip('\n') for x in val_list]
    else:
        paths = os.listdir(ori_data_path)
        with open(train_list_path, 'r') as f:
            train_list = f.readlines()
            train_list = [x.rstrip('\n') for x in train_list]
        with open(val_list_path, 'r') as f:
            val_list = f.readlines()
            val_list = [x.rstrip('\n') for x in val_list]
    paths.sort()
    # calculate statistics
    # imgs = []
    # for path in paths:
    #     img_name = '{}.nii.gz'.format(path)
    #     if Data_type == 'DSA':
    #         img_path = os.path.join(dsa_data_path, path, img_name)
    #     else:
    #         img_path = os.path.join(ori_data_path, path, img_name)
    #     img, img_affine, img_header = load_nii(img_path)
    #     img = np.transpose(img, (2, 0, 1))
    #     imgs.append(img)

    # imgs_data = np.vstack(imgs)
    dataset_train = {}
    dataset_val = {}
    for i, path in enumerate(paths):
        # if path not in deleted_datalist and path not in new_deleted_datalist:
        if path not in deleted_datalist:
            img_name = '{}.nii.gz'.format(path)
            gt_name = '{}_gt.nii.gz'.format(path)
            if Data_type == 'DSA':
                img_path = os.path.join(dsa_data_path, path, img_name)
                gt_path = os.path.join(dsa_label_path, path, gt_name)
            else:
                img_path = os.path.join(ori_data_path, path, img_name)
                gt_path = os.path.join(ori_label_path, path, gt_name)
            img, img_affine, img_header = load_nii(img_path)
            img = np.transpose(img, (2, 0, 1))
            # img = imgs[i]
            if Data_type == 'CTA':
                img = windwo_transform(img, window_width, window_center)
            gt, gt_affine, gt_header = load_nii(gt_path)
            # print('gt:', gt.shape,path)
            gt = np.transpose(gt, (2, 0, 1))
            args = np.argwhere(gt == 1)
            # print(args.shape[0], 'size of nonzero')

            if path in train_list:
                dataset_train[path] = {}
                dataset_train[path]['center'] = [[pt_center[0], pt_center[1], pt_center[2]]]
                if Data_type == 'CTA':
                    dataset_train[path]['img'] = normalize_img_after_windowtransform(img, window_center,
                                                                                 window_width,norm_type=norm_type) if norm else img
                elif Data_type == 'DSA':
                    dataset_train[path]['img'] = normalize_img(img,norm_type=norm_type) if norm else img
                dataset_train[path]['gt'] = convert_to_one_hot(gt) if one_hot else gt
                dataset_train[path]['nii_'] = [gt_affine, gt_header]
            elif path in val_list:
                dataset_val[path] = {}
                dataset_val[path]['center'] = [[pt_center[0], pt_center[1], pt_center[2]]]
                if Data_type == 'CTA':
                    dataset_val[path]['img'] = normalize_img_after_windowtransform(img, window_center,
                                                                               window_width,norm_type=norm_type) if norm else img
                elif Data_type == 'DSA':
                    dataset_val[path]['img'] = normalize_img(img,norm_type=norm_type) if norm else img
                dataset_val[path]['gt'] = convert_to_one_hot(gt) if one_hot else gt
                dataset_val[path]['nii_'] = [gt_affine, gt_header]
            else:
                # raise Exception('sample ID not in train_list nor val_list')
                continue

    print(ctime())

    return dataset_train, dataset_val

def get_all_labeled_data(norm=True, one_hot=False, window_width=700, window_center=80,Data_type = 'CTA',pt_center=(128,128,128),norm_type='z-score'):
    print(ctime())
    if Data_type == 'DSA':
        paths = os.listdir(dsa_data_path)
        delete_list = filtered_DSA_datalist
        # print(paths)
        with open(train_dsa_list_path, 'r') as f:
            train_list = f.readlines()
            train_list = [x.rstrip('\n') for x in train_list]
        with open(val_dsa_list_path, 'r') as f:
            val_list = f.readlines()
            val_list = [x.rstrip('\n') for x in val_list]
    else:
        paths = os.listdir(ori_data_path)
        delete_list = filtered_CTA_datalist
        with open(train_list_path, 'r') as f:
            train_list = f.readlines()
            train_list = [x.rstrip('\n') for x in train_list]
        with open(val_list_path, 'r') as f:
            val_list = f.readlines()
            val_list = [x.rstrip('\n') for x in val_list]
    paths.sort()
    # calculate statistics

    dataset_train = {}
    len_xs,len_ys,len_zs = [],[],[]
    for i, path in enumerate(paths):
        # if path not in deleted_datalist and path not in new_deleted_datalist:
        if path not in delete_list:
            img_name = '{}.nii.gz'.format(path)
            gt_name = '{}_gt.nii.gz'.format(path)
            if Data_type == 'DSA':
                img_path = os.path.join(dsa_data_path, path, img_name)
                gt_path = os.path.join(dsa_label_path, path, gt_name)
            else:
                img_path = os.path.join(ori_data_path, path, img_name)
                gt_path = os.path.join(ori_label_path, path, gt_name)
            img, img_affine, img_header = load_nii(img_path)
            img = np.transpose(img, (2, 0, 1))
            if Data_type == 'CTA':
                img = windwo_transform(img, window_width, window_center)
            gt, gt_affine, gt_header = load_nii(gt_path)
            # print('gt:', gt.shape,path)
            gt = np.transpose(gt, (2, 0, 1))
            args = np.argwhere(gt == 1)

            # if path in train_list:
            dataset_train[path] = {}
            dataset_train[path]['center'] = [[pt_center[0], pt_center[1], pt_center[2]]]
            if Data_type == 'CTA':
                dataset_train[path]['img'] = normalize_img_after_windowtransform(img, window_center,
                                                                             window_width,norm_type=norm_type) if norm else img
            elif Data_type == 'DSA':
                dataset_train[path]['img'] = normalize_img(img,norm_type=norm_type) if norm else img
            dataset_train[path]['gt'] = convert_to_one_hot(gt) if one_hot else gt
            dataset_train[path]['nii_'] = [gt_affine, gt_header]

    print(ctime())

    return dataset_train


def crop_data():
    ori_data_path = '/home/hci/Datasets/Aneurysm/DSA/Data'
    paths = os.listdir(ori_data_path)
    crop_data_path = '/home/hci/Datasets/Aneurysm/DSA/crop_data'
    if not os.path.exists(crop_data_path):
        os.mkdir(crop_data_path)
    for path in paths:
        img_name = '{}.nii.gz'.format(path)
        gt_name = '{}_gt.nii.gz'.format(path)
        crop_img_path = os.path.join(crop_data_path,path)
        if not os.path.exists(crop_img_path):
            os.mkdir(crop_img_path)
        img_path = os.path.join(ori_data_path, path, img_name)
        gt_path = os.path.join(ori_data_path, path, gt_name)
        img, img_affine, img_header = load_nii(img_path)
        gt, gt_affine, gt_header = load_nii(gt_path)
        # print('name:', path, img.shape, gt.shape)
        cropped_img,cropped_gt = roi_crop_data(img,gt,crop_size=[256,256,256])
        crop_imgdata_path = os.path.join(crop_img_path,'{}.nii.gz'.format(path))
        save_nii(crop_imgdata_path,cropped_img,img_affine,img_header)
        crop_gt_path = os.path.join(crop_img_path, '{}_gt.nii.gz'.format(path))
        save_nii(crop_gt_path, cropped_gt, gt_affine, gt_header)

def get_dataset(data_type,FULL_TRAINING,train_num,norm_type='z-score'):
    train_dataset, test_dataset = get_labeled_data(norm=True, one_hot=True, Data_type=data_type,norm_type=norm_type)

    # n-fold or full
    if FULL_TRAINING:
        train_dataset = dict(train_dataset.items() + test_dataset.items())
        val_dataset = train_dataset
        test_dataset = train_dataset
    else:
        random.seed(2333)
        all_keys = list(train_dataset.keys())
        random.shuffle(all_keys)
        train_keys = all_keys[:train_num]
        val_keys = all_keys[train_num:]
        new_train_dataset = {k: v for k, v in train_dataset.items() if k in train_keys}
        val_dataset = {k: v for k, v in train_dataset.items() if k in val_keys}
        train_dataset = new_train_dataset
    return train_dataset,val_dataset,test_dataset

