import torch
import torch.nn as nn
from models.Model_Utils import softmax_helper
from data_preprocessing.Data_Utils import convert_to_one_hot
import torch.nn.functional as F


class pmkl_loss(nn.Module):
    def __init__(self,eps=1.,temperature=5.):
        super(pmkl_loss, self).__init__()
        self.eps = eps
        self.temperature = temperature

    def entropy_loss(self,st_out, label, num_cls):
        for i in range(num_cls):
            labeli = label[:, i, :, :, :]
            predi = st_out[:, i, :, :, :]
            if i == 0:
                raw_loss = -1.0 * labeli * torch.log(torch.clip(predi, 0.005, 1))
            else:
                raw_loss += -1.0 * labeli * torch.log(torch.clip(predi, 0.005, 1))
        loss = raw_loss
        return loss

    def kd_loss(self,te_out, st_out, num_cls,temperature=5.):
        soft_te_out = nn.Softmax(dim=1)(te_out/temperature)
        soft_st_out = nn.Softmax(dim=1)(st_out/temperature)
        kd_entropy = self.entropy_loss(soft_st_out, soft_te_out, num_cls)

        return kd_entropy

    def contrast_loss(self,teacher_feat,student_feat,eps=1.):
        data_shape = teacher_feat.shape
        teacher_feat,student_feat = teacher_feat.unsqueeze(1),student_feat.unsqueeze(0)#(2,1,16),(1,2,16)
        # loss_criterion = torch.nn.MSELoss(reduction='none')
        #compute L2 distance matric
        loss_matric = (student_feat - teacher_feat)#(2,2,16)
        loss_matric = loss_matric * loss_matric
        n = loss_matric.shape[2]
        loss_matric = torch.sqrt(torch.sum(loss_matric,dim=2))/n
        batch_size = data_shape[0]
        loss = 0.
        for i in range(batch_size):
            for j in range(batch_size):
                if j == i:
                    loss += loss_matric[j,i]*loss_matric[j,i]
                else:
                    tempt_loss = eps - loss_matric[j,i] if eps > loss_matric[j,i] else 0
                    loss += tempt_loss*tempt_loss

        return loss

    def forward(self,teacher_out,student_out,teacher_feat,student_feat,label,contrast=True,kd=True):
        loss = []
        if kd:
            cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
            te_ce_loss = cross_entropy_loss(teacher_out,label.squeeze(1).long())
            st_ce_loss = cross_entropy_loss(student_out, label.squeeze(1).long())

            kd_mask1 = nn.ReLU()(st_ce_loss - te_ce_loss)
            pred_teacher_out = softmax_helper(teacher_out)
            te_pred_compact = torch.argmax(pred_teacher_out,dim=1)
            kd_mask2 = torch.eq(te_pred_compact,label)
            kd_mask = kd_mask2*kd_mask1

            kd_pixel_loss = self.kd_loss(teacher_out,student_out,num_cls=2,temperature=self.temperature)
            kd_ce_loss = torch.sum(kd_pixel_loss*kd_mask)/torch.sum(kd_mask)
            loss.append(kd_ce_loss)

        if contrast:
            contrast_loss_ = self.contrast_loss(teacher_feat,student_feat,eps=self.eps)
            loss.append(contrast_loss_)

        return loss

class kd_loss(nn.Module):
    def __init__(self,eps=1.,temperature=5.,loss="entropy"):
        super(kd_loss, self).__init__()
        self.eps = eps
        self.temperature = temperature
        self.loss_type = loss

    def entropy_loss(self,st_out, label, num_cls):
        for i in range(num_cls):
            labeli = label[:, i, :, :, :]
            predi = st_out[:, i, :, :, :]
            if i == 0:
                raw_loss = -1.0 * labeli * torch.log(torch.clip(predi, 0.005, 1))
            else:
                raw_loss += -1.0 * labeli * torch.log(torch.clip(predi, 0.005, 1))
        loss = raw_loss
        return loss

    def mse_consistency_loss(self, st_out, label, num_cls):
        for i in range(num_cls):
            labeli = label[:, i, :, :, :]
            predi = st_out[:, i, :, :, :]
            loss_matric = labeli - predi
            loss_matric = loss_matric * loss_matric
            # loss_matric = torch.sqrt(torch.mean(loss_matric, dim=2))
            if i == 0:

                raw_loss = loss_matric
            else:
                raw_loss += loss_matric
        loss = raw_loss
        return loss

    def kd_loss(self,te_out, st_out, num_cls,temperature=5.,loss="entropy"):
        soft_te_out = nn.Softmax(dim=1)(te_out/temperature)
        soft_st_out = nn.Softmax(dim=1)(st_out/temperature)
        if loss == "entropy":
            kd_entropy = self.entropy_loss(soft_st_out, soft_te_out, num_cls)
            return kd_entropy
        else:
            mse = self.mse_consistency_loss(soft_st_out, soft_te_out, num_cls)
            return mse
        # print('kd_entropy:',torch.sum(torch.isnan(kd_entropy)))
        # return kd_entropy


    def forward(self,teacher_out,student_out,label):
        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        te_ce_loss = cross_entropy_loss(teacher_out,label.squeeze(1).long())
        st_ce_loss = cross_entropy_loss(student_out, label.squeeze(1).long())
        # print('student_out',torch.sum(torch.isnan(student_out)))
        # print('shape:',te_ce_loss.shape,st_ce_loss.shape,torch.sum(torch.isnan(te_ce_loss)))
        kd_mask1 = nn.ReLU()(st_ce_loss - te_ce_loss)
        pred_teacher_out = softmax_helper(teacher_out)
        te_pred_compact = torch.argmax(pred_teacher_out,dim=1)
        kd_mask2 = torch.eq(te_pred_compact,label)
        kd_mask = kd_mask2*kd_mask1#stop gradient?

        kd_pixel_loss = self.kd_loss(teacher_out,student_out,num_cls=2,temperature=self.temperature,loss=self.loss_type)
        kd_ce_loss = torch.sum(kd_pixel_loss*kd_mask)/torch.sum(kd_mask)

        return kd_ce_loss

# data = torch.rand((2,2,12,12))
# data = torch.sum(data,dim=[0,2,3])
# print(data.shape)
# soft = F.softmax(data)
# print(soft.shape)
# soft = torch.softmax(data,dim=0)
# print(soft.shape)

class DsbnKdLoss(nn.Module):
    def __init__(self,eps=1e-6,temperature=2.,):
        super(DsbnKdLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self,src_logits,trg_logits,src_gt,trg_gt):
        # src_logits = torch.tensor([1,2,3])
        b,c,h,w,z = src_logits.shape
        if src_gt.shape != src_logits.shape:
            src_onehot,trg_onehot = torch.zeros_like(src_logits),torch.zeros_like(src_logits)
            if len(src_gt) == 4:
                src_gt = src_gt.unsqueeze(1)
                trg_gt = trg_gt.unsqueeze(1)
            src_onehot = src_onehot.scatter_(1,src_gt.type(torch.int64),1.)
            trg_onehot = trg_onehot.scatter_(1, trg_gt.type(torch.int64), 1.)
            src_gt,trg_gt = src_onehot,trg_onehot


        kdloss = 0.
        for i in range(c):
            s_mask = src_gt[:,i].unsqueeze(1).repeat((1,c,1,1,1))
            s_logits_mask_out = src_logits*s_mask
            s_logits_avg = torch.sum(s_logits_mask_out,dim=[0,2,3,4])/(torch.sum(src_gt[:,i])+self.eps)
            s_soft_prob = F.softmax(s_logits_avg/self.temperature)

            t_mask = trg_gt[:, i].unsqueeze(1).repeat((1, c, 1, 1, 1))
            t_logits_mask_out = trg_logits * t_mask
            t_logits_avg = torch.sum(t_logits_mask_out, dim=[0, 2, 3, 4]) / (torch.sum(trg_gt[:, i]) + self.eps)
            t_soft_prob = F.softmax(t_logits_avg / self.temperature)

            loss = torch.sum(s_soft_prob*torch.log(s_soft_prob/t_soft_prob)) + torch.sum(t_soft_prob*torch.log(t_soft_prob/s_soft_prob))
            kdloss += loss/2
        kdloss = kdloss/c
        return kdloss



# import numpy as np
# import time
# data = np.random.rand(128,128,128)
# ta = time.time()
# fd = np.fft.fftn(data)
# tb = time.time()
# print(tb-ta)


# data1 = torch.range(0,4).unsqueeze(0)
# data2 = data1.t()
# print(data2.shape,data1.shape)
# print(data1-data2)
# data1 = torch.rand(3,4)
# data2 = torch.rand(3,4)
# print(data1)
# print(data2)
# print(torch.div(data1,data2))
# print(data2.sum(0))
# data1 = data1.unsqueeze(0)
# data2 = data2.unsqueeze(1)
# print(data1 - data2)

