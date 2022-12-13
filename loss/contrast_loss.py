import torch.nn as nn
import torch
import torch.nn.functional as F
from Models.Multi_Modal_Seg.model_utils import Upsample_3D
from einops import rearrange
from Config.Data_Config import ori_data_path,ori_label_path
import os
from Data_Preprocessing.Data_Utils import split_data
import matplotlib.pyplot as plt
from Data_Augmentation import get_default_augmentation, default_3D_augmentation_params
from Loss.Dice_loss import SoftDiceLoss, SoftDiceLossSquared, DC_and_CE_loss
from Baseline.UNet import Generic_UNet
from Data_Reader_CADA import get_labeled_data,mutithread_get_data
from Data_Generator import DataGenerator3D
from Data_Augmentation import get_default_augmentation
from multimodal_cutmix import get_mix_batch
from batchgenerators.augmentations.utils import resize_segmentation
data_path = os.path.join(ori_data_path,'CaoYanYan','CaoYanYan.nii.gz')
label_path = os.path.join(ori_label_path,'CaoYanYan','CaoYanYan_gt.nii.gz')

# img,_,__ = load_nii(data_path)
# label,_,__ = load_nii(label_path)


class ContrastAllignmentLoss(nn.Module):
    def __init__(self,num_samples):
        super(ContrastAllignmentLoss, self).__init__()
        self.num_samples = num_samples

    def _compute_sim(self,src_feat,trg_feat,temparature=0.1):
        src_feat,trg_feat = src_feat.flatten(2),trg_feat.flatten(2)
        # x,y = x[x!=0],y[y!=0]
        src_feat = src_feat.permute(0,2,1)
        src_norm,trg_norm = F.normalize(src_feat,dim=1),F.normalize(trg_feat,dim=2)
        sim = torch.bmm(src_norm,trg_norm)/temparature
        return sim

    def sample_pos(self,target_num,target_index):
        if target_num > self.num_samples*2/3:
            target_num = self.num_samples * 2//3
            select_index = torch.randperm(target_index.shape[1])[:target_num].cuda()
            target_index = torch.index_select(target_index, dim=1, index=select_index)
            return target_index,target_num
        return target_index,target_num

    def _anchor_samples(self,src,trg,src_mask,trg_mask):
        b,c,feat_dim = src.shape
        #mask:1,dim
        #select target pixel according to mask
        src_mask,trg_mask = src_mask.squeeze(),trg_mask.squeeze()
        src_target_index,trg_target_index = src_mask.nonzero().permute(1,0),trg_mask.nonzero().permute(1,0)
        #number of the target pixel
        src_target_num,trg_target_num = src_target_index.shape[1],trg_target_index.shape[1]
        #in case the target number is too large
        sampled_num = self.num_samples
        # src_target_num, trg_target_num = tuple(map(lambda x:sampled_num*2//3 if x> sampled_num*2/3 else x,[src_target_num, trg_target_num]))
        src_target_index, src_target_num = self.sample_pos(src_target_num,src_target_index)
        trg_target_index, trg_target_num = self.sample_pos(trg_target_num, trg_target_index)
        # print('trg:',trg_target_index.shape, trg_target_num)
        #number of the background pixel needed to sample
        src_no_sim_num,trg_no_sim_num = self.num_samples - src_target_num,self.num_samples - trg_target_num
        src_no_sim_index,trg_no_sim_index = (1-src_mask).nonzero().permute(1,0),(1-trg_mask).nonzero().permute(1,0)
        #random sample background pixel
        # print(src_no_sim_index.shape)
        src_no_sim_sampled_index,trg_no_sim_sampled_index = torch.randperm(src_no_sim_index.shape[1])[:src_no_sim_num].cuda(),\
                                                            torch.randperm(trg_no_sim_index.shape[1])[:trg_no_sim_num].cuda()
        # print('no_sim_num:',src_no_sim_sampled_index.shape,src_no_sim_num,self.num_samples)
        src_no_sim_sampled_index,trg_no_sim_sampled_index = torch.index_select(src_no_sim_index,dim=1,index=src_no_sim_sampled_index),\
                                torch.index_select(trg_no_sim_index,dim=1,index=trg_no_sim_sampled_index)
        # print(src_no_sim_index.shape,src_target_index.shape, src_no_sim_sampled_index.shape)
        src_sampled_index, trg_sampled_index = torch.cat((src_target_index, src_no_sim_sampled_index), dim=1), \
                                               torch.cat((trg_target_index, trg_no_sim_sampled_index), dim=1)
        # src_target, trg_target = torch.index_select(src, dim=2, index=src_target_index.squeeze(0)), torch.index_select(trg, dim=2, index=trg_target_index.squeeze(0))
        # src_notarget, trg_notarget = torch.index_select(src, dim=2, index=src_no_sim_sampled_index.squeeze(0)), torch.index_select(trg, dim=2, index=trg_no_sim_sampled_index.squeeze(0))
        src_sampled,trg_sampled = torch.index_select(src,dim=2,index=src_sampled_index.squeeze()),\
                                  torch.index_select(trg,dim=2,index=trg_sampled_index.squeeze())
        src_sampled_mask,trg_sampled_mask = src_mask[src_sampled_index],trg_mask[trg_sampled_index]
        src_sampled_mask_mat,trg_sampled_mask_mat = src_sampled_mask.expand(self.num_samples,self.num_samples),\
                                                    trg_sampled_mask.expand(self.num_samples,self.num_samples).t()
        # mask = torch.eq(src_sampled_mask_mat,trg_sampled_mask_mat.permute(1,0)).float().cuda()
        #select foreground samples in both src domain and trg domain
        return src_sampled,trg_sampled,src_sampled_mask_mat,trg_sampled_mask_mat
        # sim = mask *

    def _contrastive(self,src_feat,trg_feat,src_label,trg_label,temparature=0.1):
        '''
        :param src_feat:
        :param trg_feat: b,c,D,H,W
        :param src_label: b,1,D,H,W
        :param trg_label:
        :param temparature:
        :return:
        '''
        B,C,D,H,W = src_feat.shape
        src_feat, trg_feat, src_label, trg_label = src_feat.flatten(2), trg_feat.flatten(2), src_label.flatten(1), trg_label.flatten(1)
        src_feat,trg_feat = tuple(map(lambda x: rearrange(x,'B C T -> C (B T)',C=C,B=B).unsqueeze(0),[src_feat,trg_feat]))
        src_label, trg_label = tuple(map(lambda x: rearrange(x.unsqueeze(0),'d B T -> d (B T)',d=1,B=B),[src_label, trg_label]))
        num_classes = torch.unique(src_label)
        loss = 0.
        for cls in range(1, len(num_classes)):
            src_mask = src_label * (src_label == cls)
            trg_mask = trg_label * (trg_label == cls)
            src_sampled,trg_sampled,src_sampled_mask_mat,trg_sampled_mask_mat = self._anchor_samples(src_feat,trg_feat,src_mask,trg_mask)
            loss += self._compute_loss(src_sampled,trg_sampled,src_sampled_mask_mat,trg_sampled_mask_mat,temparature)

        return loss

    def _compute_loss(self,src_sampled,trg_sampled,src_sampled_mask_mat,trg_sampled_mask_mat,temparature=0.1):
        mask = ((src_sampled_mask_mat > 0) & (trg_sampled_mask_mat > 0)).float().cuda()
        #select negative samples
        neg_mask = (((src_sampled_mask_mat == 0) & (trg_sampled_mask_mat > 0))|((src_sampled_mask_mat > 0) & (trg_sampled_mask_mat == 0))).float().cuda()
        sim_matrix = self._compute_sim(src_sampled, trg_sampled,temparature)
        sim_max,_ = torch.max(sim_matrix,dim=2, keepdim=True)
        logits = sim_matrix - sim_max.detach()
        #compute 分母
        neg_logits = torch.exp(sim_max)*neg_mask
        neg_logits_sum = torch.sum(neg_logits,dim=2,keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits_sum)
        # print('log_prob',log_prob.shape,log_prob)
        # mean_log_prob_pos = torch.sum(mask * log_prob,dim=2)/mask.sum(1)
        mean_log_prob_pos = torch.sum(mask * log_prob) / mask.sum()
        # print(mean_log_prob_pos)
        # print(mean_log_prob_pos.shape,torch.sum(mask * log_prob,dim=2).shape,mask.shape,mask.sum(1))
        # loss = -mean_log_prob_pos.mean()
        loss = -mean_log_prob_pos
        # print(loss)
        return loss

    def forward(self,src_feat,trg_feat,src_label,trg_label,temparature=10):
        # ori_shape = src_label.shape
        # new_shape = src_feat.shape[2:]
        # src_downsampled_label = resize_segmentation(src_label,new_shape)
        # trg_downsampled_label = resize_segmentation(trg_label, new_shape)
        # src_upsampled = nn.functional.interpolate(src_feat,size=ori_shape[2:],mode='trilinear')
        # trg_upsampled = nn.functional.interpolate(trg_feat, size=ori_shape[2:], mode='trilinear')
        return self._contrastive(src_feat,trg_feat,src_label,trg_label,temparature)

# import time
# print('start',time.ctime())
# data1,data2 = torch.rand(2,32,400).cuda(),torch.rand(2,400,32).cuda()
# data3 = torch.bmm(data2,data1)
# # data3 = data2@data1
# # data3 = torch.einsum('b c t,b t c -> b c c',data2,data1)
# print('end',time.ctime())
# print(data3.shape)
#
import random
train_num = 36
train_dataset, test_dataset = get_labeled_data(norm=True, one_hot=True,Data_type="CTA")
# src_train_dataset, src_test_dataset = get_labeled_data(norm=True, one_hot=True,Data_type='DSA')
dataset = dict(list(train_dataset.items()) + list(test_dataset.items()))
random.seed(2333)
splits = split_data(list(dataset.keys()), K=5, shuffle=True)
train_dataset = {k:v for k,v in dataset.items() if k in splits[0]['train']}
test_dataset = {k:v for k,v in dataset.items() if k in splits[0]['val']}
print('train:',[k for k,v in train_dataset.items()])
print('test:',[k for k,v in test_dataset.items()])
# val_dataset = get_labeled_data(norm=True, one_hot=True)
# n-fold or full

random.seed(2333)
all_keys = list(train_dataset.keys())
random.shuffle(all_keys)
train_keys = all_keys[:train_num]
val_keys = all_keys[train_num:]
new_train_dataset = {k: v for k, v in train_dataset.items() if k in train_keys}
val_dataset = {k: v for k, v in train_dataset.items() if k in val_keys}
train_dataset = new_train_dataset

print('train set: {}'.format(len(train_dataset.keys())))
print('val set: {}'.format(len(val_dataset.keys())))
print('test set: {}'.format(len(test_dataset.keys())))
#
train_dataset_1 = train_dataset
train_loader = DataGenerator3D(train_dataset, patch_size=(128,128,128), batch_size=2)
train_gen, _ = get_default_augmentation(train_loader, None, (128,128,128), deep_supervision_scales=[[1,1,1],[0.5,0.5,0.5]],
                                        params=default_3D_augmentation_params)
# src_train_loader = DataGenerator3D(src_train_dataset, patch_size=(128,128,128), batch_size=2)
# src_train_gen, _ = get_default_augmentation(src_train_loader, None, (128,128,128), params=default_3D_augmentation_params)

import numpy as np

def resize(input,new_size):
    out = F.interpolate(input=input, size=new_size, mode='trilinear', align_corners=True)
    return out

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-3,-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-3,-2, -1) )

    _,__, d,h, w = a_src.shape
    b = (  np.floor(np.amin((d,h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    c_d = np.floor(d / 2.0).astype(int)

    d1 = c_d -b
    d2 = c_d + b +1
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,:,d1:d2,h1:h2,w1:w2] = a_trg[:,:,d1:d2,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-3,-2, -1) )
    return a_src

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target

    fft_src_np = np.fft.fftn( src_img_np, axes=(-3,-2, -1) )
    fft_trg_np = np.fft.fftn( trg_img_np, axes=(-3,-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifftn( fft_src_, axes=(-3,-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

def transform(train_img,train_label,scaled_label):
    if not isinstance(train_img, torch.Tensor):
        train_img = torch.from_numpy(train_img).float()
    if not isinstance(train_label, torch.Tensor):
        train_label = torch.from_numpy(train_label).float()
        scaled_label = torch.from_numpy(scaled_label).float()
    if torch.cuda.is_available():
        train_img = train_img.cuda(non_blocking=True)
        train_label = train_label.cuda(non_blocking=True)
        scaled_label = scaled_label.cuda(non_blocking=True)
    return train_img,train_label,scaled_label
#
# #
from region_contrast import generate_batch_regions
print('............')
for i in range(100):
    train_batch = next(train_gen)
    train_img = train_batch['data']
    train_label = train_batch['target']
    train_label, scaled_label = train_label[0], train_label[1]
    sample_regions = generate_batch_regions(train_img,train_label)
# src_train_batch = next(src_train_gen)
# src_train_img = src_train_batch['data']
# src_train_label = src_train_batch['target']
# # lam = np.random.uniform(0.4, 0.8)
# lam = np.random.beta(1, 1)
# print(lam)
# mix_trg_img, mix_trg_label, bboxs = get_mix_batch(src_train_img, train_img, src_train_label, train_label, lam=lam)
# [bbx1, bby1,bbz1, bbx2, bby2,bbz2,lt_x,rb_x,lt_y,rb_y,lt_s,rb_s] = bboxs[0]
# # print(train_img.shape,src_train_label.shape)
# # print(np.unique(src_train_label))
# args = np.argwhere(mix_trg_label == 1)
# # print('args:',args.shape)
# min_x, max_x, min_y, max_y, min_z, max_z = min(args[2,:]), max(args[2,:]), min(args[:, 3]), max(args[:, 3]), min(args[:, 4]), max(args[:, 4])
#
# # src_to_trg = FDA_source_to_target_np(src_train_img,train_img,0.25)
# import matplotlib.pyplot as plt
# center = (bbx1+bbx2)//2
# for i in range(20):
#     print(np.random.randint(128))
# for i in range(bbx1,bbx2):
#     print(i)
#     if(len(np.unique(mix_trg_label[0,0,i,:,:]))>1):
#         plt.subplot(2,2,1)
#         plt.imshow(src_train_img[0,0,i,:,:],cmap='gray')
#         plt.subplot(2, 2, 2)
#         plt.imshow(train_img[0, 0, i, :, :], cmap='gray')
#         plt.subplot(2, 2, 3)
#         plt.imshow(mix_trg_img[0, 0, i, :, :], cmap='gray')
#         plt.subplot(2, 2, 4)
#         # print()
#         plt.imshow(mix_trg_label[0, 0,i, :, :]*512., cmap='gray')
#         plt.show()


# print(train_img)
# train_img,train_label,scaled_label = transform(train_img,train_label,scaled_label)
# print(torch.sum(train_label == 1), torch.sum(scaled_label == 1))
# src_batch = next(train_gen)
# src_img = src_batch['data']
# src_label = src_batch['target']
# src_label, src_scaled_label = src_label[0], src_label[1]
# src_img ,src_label, src_scaled_label = transform(src_img ,src_label, src_scaled_label)
# print(torch.sum(src_label == 1), torch.sum(src_scaled_label == 1))
# src_img,train_img = resize(src_img,(64,64,64)),resize(train_img,(64,64,64))
# print(src_img.shape,train_img.shape)
# criterion = ContrastAllignmentLoss(num_samples=400)
# loss = criterion(src_img,train_img,src_scaled_label,scaled_label,2)
# print(loss)
# loss = criterion(src_img,train_img,src_scaled_label,scaled_label,5)
# print(loss)
# loss = criterion(src_img,train_img,src_scaled_label,scaled_label,10)
# print(loss)
# for i in range(20):
#     train_batch = next(train_gen)
#     train_img = train_batch['data']
#     train_label = train_batch['target']
#     train_label,scaled_label = train_label[0],train_label[1]
#
#     # print(train_label.shape,train_img.shape,scaled_label.shape)
#     print(torch.sum(train_label==1),torch.sum(scaled_label==1))
# data = torch.rand(2,256,8,8,8)
# model = Upsample_3D(size=(32,32,32),mode='trilinear')
# out = model(data)
# print(out.shape)

# def consistency_semantic_loss(src_feat,trg_feat,src_label,trg_label):
#

#
# import torch
#
# import torch.nn.functional as F
# T = 0.5  #温度参数T
#
# label = torch.tensor([1,0,1,0,1])
# n = label.shape[0]  # batch
# #假设我们的输入是5 * 3  5是batch，3是句长
# representations = torch.tensor([[1, 2, 3],[1.2, 2.2, 3.3],
#
#                                 [1.3, 2.3, 4.3],[1.5, 2.6, 3.9],
#
#                                 [5.1, 2.1, 3.4]])
# print(label.expand(n,n))
# #这步得到它的相似度矩阵
# similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
# print(similarity_matrix.shape,similarity_matrix)
# print((label.expand(n, n).eq(label.expand(n, n).t())))#
# #这步得到它的label矩阵，相同label的位置为1,每个样本与其他位置样本的类别相同与否,第一行的每二个数值表示第一个样本与第二个样本的类别是否相同，
# #第一行的第三个数值表示第三个样本与第一个样本是否相同
# mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
# #这步得到它的不同类的矩阵，不同类的位置为1
# mask_no_sim = torch.ones_like(mask) - mask
# #这步产生一个对角线全为0的，其他位置为1的矩阵
# mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n )
# #这步给相似度矩阵求exp,并且除以温度参数T
# similarity_matrix = torch.exp(similarity_matrix/T)
# #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
# similarity_matrix = similarity_matrix*mask_dui_jiao_0
# #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
# sim = mask*similarity_matrix
# #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
# no_sim = similarity_matrix - sim
# #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还一个与分子相同的那个相似度，后面会加上)
# no_sim_sum = torch.sum(no_sim , dim=1)
# print(no_sim_sum.shape,no_sim_sum)
# '''
# 将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
# 至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
# 每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
# '''
# no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
# print(no_sim_sum_expend.shape,no_sim_sum_expend)
# sim_sum  = sim + no_sim_sum_expend
# print(sim_sum)
# print(sim)
# loss = torch.div(sim , sim_sum)
# print(loss)
# '''
# 由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
# 全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
# '''
# loss = mask_no_sim + loss + torch.eye(n, n )
# #接下来就是算一个批次中的loss了
# loss = -torch.log(loss)  #求-log
# loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n
# print(loss)  #0.9821
# #最后一步也可以写为
# # loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))
# data = torch.rand(2,3,4)
# data1 = torch.rand(2,3,4)
# data2 = torch.rand(4)
# print(data,data2)
# index = (data>0.5).nonzero()
# # print(index,index.shape)
# print(data1)
# print((data>0.5)|(data1>0.5))
# mask = data2>0.5
# index1 = mask.nonzero()
# print('mask:',mask[index1])
# # print(data[:,:,])
# # print(torch.masked_select(data,mask))
# # index1 = torch.tensor([[0,0,]])
# print(index1)
# result = torch.index_select(data,dim=2,index=index1.squeeze(1))
# print(result,result.shape)
# lis = [1,2,3]
# res,_,__ = tuple(map(lambda x: x+1,lis))
# print(res)

