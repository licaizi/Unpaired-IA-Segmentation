import numpy as np
import random
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from utils.region_contrast import mine_hard_negsample,sample_hard_neg
import torch
import torch.nn.functional as F

def contrast_loss(sample_regions,num_keys=8,temperature=5.,mine_hard=False,hard_samples=8,broad_cast=False):
    n,out = sample_regions.shape
    sample_regions = sample_regions.view(n//(num_keys+2),num_keys+2,out)
    anchor,query,keys = sample_regions[:,0],sample_regions[:,1],sample_regions[:,2:]#b,k,c
    if broad_cast:
        b,k,c = keys.shape
        keys = keys.reshape(1,b*k,c)
        keys = keys.repeat(b,1,1)
    if mine_hard:
        keys = mine_hard_negsample(anchor,keys,hard_num=hard_samples)
    # print('shape:',keys.shape,query.shape)
    all_feat = torch.cat([query.unsqueeze(1),keys],dim=1)
    logits = torch.cosine_similarity(anchor.unsqueeze(1),all_feat,dim=2)
    logits /= temperature
    labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
    ce_loss = CrossEntropyLoss().cuda()
    return ce_loss(logits,labels)

def contrast_loss(sample_regions,num_keys=8,temperature=5.,mine_hard=False,hard_samples=8,broad_cast=False):
    n,out = sample_regions.shape
    sample_regions = sample_regions.view(n//(num_keys+2),num_keys+2,out)
    anchor,query,keys = sample_regions[:,0],sample_regions[:,1],sample_regions[:,2:]#b,k,c
    if broad_cast:
        b,k,c = keys.shape
        keys = keys.reshape(1,b*k,c)
        keys = keys.repeat(b,1,1)
    if mine_hard:
        keys = mine_hard_negsample(anchor,keys,hard_num=hard_samples)
    # print('shape:',keys.shape,query.shape)
    all_feat = torch.cat([query.unsqueeze(1),keys],dim=1)
    logits = torch.cosine_similarity(anchor.unsqueeze(1),all_feat,dim=2)
    logits /= temperature
    labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
    ce_loss = CrossEntropyLoss().cuda()
    return ce_loss(logits,labels)

def self_contrast_loss(trg_regions,num_keys=8,temperature=5):
    n, out = trg_regions.shape
    trg_regions = trg_regions.view(n // (num_keys + 2), num_keys + 2, out)
    # trg_regions = trg_regions.view(n // (num_keys + 3), num_keys + 3, out)
    # src_anchor, trg_query, self_pos, trg_keys = src_regions[:, 0], src_regions[:, 1], src_regions[:, 2], src_regions[:, 3:]  # b,k,c
    trg_anchor, trg_query, trg_keys = trg_regions[:, 0], trg_regions[:, 1], trg_regions[:, 2:]
    # print('shape:',keys.shape,query.shape)
    all_feat = torch.cat([trg_query.unsqueeze(1), trg_keys], dim=1)
    # self_feat = torch.cat([self_pos.unsqueeze(1), trg_keys], dim=1)
    logits = torch.cosine_similarity(trg_anchor.unsqueeze(1), all_feat, dim=2)
    logits /= temperature
    # self_logits = torch.cosine_similarity(trg_query.unsqueeze(1), self_feat, dim=2)
    # self_logits /= temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    ce_loss = CrossEntropyLoss().cuda()
    return ce_loss(logits,labels)

def self_other_multti_size_contrast_loss(src_regions,trg_regions,num_keys=8,temperature=5.,mine_hard=False,hard_samples=8):
    n,out = src_regions.shape
    src_regions = src_regions.view(n//(num_keys+4),num_keys+4,out)
    trg_regions = trg_regions.view(n // (num_keys + 4), num_keys + 4, out)
    src_anchor,self_pos,trg_query,trg_pos2,trg_keys = src_regions[:,0],src_regions[:,1],src_regions[:,2],src_regions[:,3],src_regions[:,3:]#b,k,c
    trg_anchor,src_query0,src_query,src_query2,src_keys = trg_regions[:,0],trg_regions[:,1],trg_regions[:,2],trg_regions[:,3],trg_regions[:,3:]
    if mine_hard:
        trg_keys = sample_hard_neg(trg_anchor,trg_keys,hard_num=hard_samples,temperature=temperature)
        src_keys = sample_hard_neg(trg_anchor,src_keys,hard_num=hard_samples,temperature=temperature)
    # print('shape:',keys.shape,query.shape)
    all_feat = torch.cat([src_query0.unsqueeze(1),src_query.unsqueeze(1),src_query2.unsqueeze(1),src_keys],dim=1)
    self_feat = torch.cat([self_pos.unsqueeze(1),trg_pos2.unsqueeze(1),trg_keys],dim=1)
    logits = torch.cosine_similarity(trg_anchor.unsqueeze(1),all_feat,dim=2)
    logits /= temperature
    self_logits = torch.cosine_similarity(trg_query.unsqueeze(1), self_feat, dim=2)
    self_logits /= temperature
    self_logits0 = torch.cat([self_logits[:,0].unsqueeze(1),self_logits[:,2:]],dim=1)
    labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
    ce_loss = CrossEntropyLoss().cuda()
    logits0 = torch.cat([logits[:,0].unsqueeze(1),logits[:,3:]],dim=1)
    logits1 = torch.cat([logits[:, 1].unsqueeze(1), logits[:, 3:]], dim=1)
    cross_loss = ce_loss(logits0,labels) +ce_loss(logits[:,2:],labels)+ce_loss(logits1,labels)
    self_loss = ce_loss(self_logits0,labels) + ce_loss(self_logits[:,1:],labels)
    return cross_loss/3,self_loss/2

def LMI_onemod_contrast_loss(trg_embed,trg_regions,num_keys=100,temperature=0.1):
    '''

    :param trg_embed: n,dim
    :param src_embed: n,dim
    :param src_regions: n*(key+3),dim
    :param trg_regions: n*(key+3),dim
    :param num_keys:
    :param temperature:
    :return:
    '''
    n, out = trg_regions.shape
    # src_regions = src_regions.view(n // (num_keys + 3), num_keys + 3, out)
    trg_regions = trg_regions.view(n // (num_keys + 2), num_keys + 2, out)
    # src_anchor, trg_query, self_pos, trg_keys = src_regions[:, 0], src_regions[:, 1], src_regions[:, 2], src_regions[:, 3:]  # b,k,c
    # trg_anchor, src_query, src_keys = trg_regions[:, 0], trg_regions[:, 1], trg_regions[:, 3:]
    trg_l = trg_regions[:,1:]#n,(2+key),dim
    # src_l = trg_regions[:,1:]
    trg_lmi_loss = in_batch_g2l_loss(trg_l,trg_embed,temperature)
    return trg_lmi_loss


def multi_mod_contrast_loss(src_sample_regions,trg_sample_regions,num_keys=8,temperature=5.,mine_hard=False,hard_samples=8,broad_cast=False):
    n, out = src_sample_regions.shape
    src_sample_regions = src_sample_regions.view(n // (num_keys + 1), num_keys + 1, out)
    src_anchor, src_keys = src_sample_regions[:, 0], src_sample_regions[:, 1:]  # b,k,c
    trg_sample_regions = trg_sample_regions.view(n // (num_keys + 1), num_keys + 1, out)
    trg_anchor, trg_keys = trg_sample_regions[:, 0], trg_sample_regions[:, 1:]  # b,k,c
    # if mine_hard:
    #     keys = mine_hard_negsample(anchor, keys, hard_num=hard_samples)
    # print('shape:',keys.shape,query.shape)
    src_all_feat = torch.cat([src_anchor.unsqueeze(1), src_keys], dim=1)
    trg_all_feat = torch.cat([trg_anchor.unsqueeze(1), trg_keys], dim=1)
    trg_logits = torch.cosine_similarity(trg_anchor.unsqueeze(1), src_all_feat, dim=2)
    src_logits = torch.cosine_similarity(src_anchor.unsqueeze(1), trg_all_feat, dim=2)
    src_logits /= temperature
    trg_logits /= temperature
    labels = torch.zeros(src_logits.shape[0], dtype=torch.long).cuda()
    ce_loss = CrossEntropyLoss().cuda()
    return ce_loss(trg_logits, labels),ce_loss(src_logits, labels)

def contrast_negmin_loss(sample_regions,num_keys=8,temperature=5.,tau_plus=0.05,beta=0.5,broad_cast=False):
    n,out = sample_regions.shape
    sample_regions = sample_regions.view(n//(num_keys+2),num_keys+2,out)
    anchor,query,keys = sample_regions[:,0],sample_regions[:,1],sample_regions[:,2:]#b,k,c
    if broad_cast:
        b,k,c = keys.shape
        keys = keys.reshape(1,b*k,c)
        keys = keys.repeat(b,1,1)
    # print('shape:',keys.shape,query.shape)
    pos = torch.exp(torch.cosine_similarity(anchor,query,dim=1)/temperature)
    # pos = torch.cat([pos,pos],dim=0)
    neg = torch.exp(torch.cosine_similarity(anchor.unsqueeze(1),keys,dim=2)/temperature)
    N = neg.shape[1]
    imp = (beta*neg.log()).exp()
    reweight_neg = (imp*neg).sum(dim=-1)/imp.mean(dim=-1)
    Ng = (-tau_plus*N*pos +reweight_neg)/(1-tau_plus)
    Ng = torch.clamp(Ng,min=N*np.e**(1/temperature))
    loss = (-torch.log(pos/(pos+Ng))).mean()
    return loss

def multi_contrast_loss(sample_regions,num_keys=8,temperature=5.,mine_hard=False,hard_samples=8,broad_cast=False):
    #cross-contrast:negative keys from self modality

    n,out = sample_regions.shape
    sample_regions = sample_regions.view(n//(num_keys+3),num_keys+3,out)
    anchor,query,self_pos,keys = sample_regions[:,0],sample_regions[:,1],sample_regions[:,2],sample_regions[:,3:]#b,k,c
    if broad_cast:
        b,k,c = keys.shape
        keys = keys.reshape(1,b*k,c)
        keys = keys.repeat(b,1,1)
    if mine_hard:
        keys = mine_hard_negsample(anchor,keys,hard_num=hard_samples)
    # print('shape:',keys.shape,query.shape)
    all_feat = torch.cat([query.unsqueeze(1),keys],dim=1)
    self_feat = torch.cat([self_pos.unsqueeze(1),keys],dim=1)
    logits = torch.cosine_similarity(anchor.unsqueeze(1),all_feat,dim=2)
    logits /= temperature
    self_logits = torch.cosine_similarity(query.unsqueeze(1), self_feat, dim=2)
    self_logits /= temperature
    labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
    ce_loss = CrossEntropyLoss().cuda()
    return ce_loss(logits,labels),ce_loss(self_logits,labels)

def self_dif_other_contrast_loss(src_regions,trg_regions,num_keys=8,temperature=5.,mine_hard=False,hard_samples=8):
    n,out = src_regions.shape
    src_regions = src_regions.view(n//(num_keys+2),num_keys+2,out)
    trg_regions = trg_regions.view(n // (num_keys + 2), num_keys + 2, out)
    src_anchor,src_pos,src_keys = src_regions[:,0],src_regions[:,1],src_regions[:,2:]#b,k,c
    trg_anchor,trg_pos,trg_keys = trg_regions[:,0],trg_regions[:,1],trg_regions[:,2:]
    if mine_hard:
        trg_keys = sample_hard_neg(trg_anchor,trg_keys,hard_num=hard_samples)
        src_keys = sample_hard_neg(src_anchor,src_keys,hard_num=hard_samples)
    # print('shape:',keys.shape,query.shape)
    all_feat = torch.cat([src_anchor.unsqueeze(1),src_keys],dim=1)
    self_feat = torch.cat([trg_pos.unsqueeze(1),trg_keys],dim=1)
    logits = torch.cosine_similarity(trg_anchor.unsqueeze(1),all_feat,dim=2)
    logits /= temperature
    self_logits = torch.cosine_similarity(trg_anchor.unsqueeze(1), self_feat, dim=2)
    self_logits /= temperature
    labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
    ce_loss = CrossEntropyLoss().cuda()
    return ce_loss(logits,labels),ce_loss(self_logits,labels)

def intra_inter_contrast_loss(src_regions,trg_regions,num_keys=8,temperature=5.,mine_hard=False,hard_samples=8):
    #cross-contrast:negative keys from other modality
    n,out = src_regions.shape
    src_regions = src_regions.view(n//(num_keys+3),num_keys+3,out)
    trg_regions = trg_regions.view(n // (num_keys + 3), num_keys + 3, out)
    src_anchor,trg_query,self_pos,trg_keys = src_regions[:,0],src_regions[:,1],src_regions[:,2],src_regions[:,3:]#b,k,c
    trg_anchor,src_query,src_keys = trg_regions[:,0],trg_regions[:,1],trg_regions[:,3:]

    if mine_hard:
        trg_keys = sample_hard_neg(trg_anchor, trg_keys, hard_num=hard_samples,temperature=temperature)
        src_keys = sample_hard_neg(trg_anchor, src_keys, hard_num=hard_samples)
    all_feat = torch.cat([src_query.unsqueeze(1),src_keys],dim=1)
    self_feat = torch.cat([self_pos.unsqueeze(1),trg_keys],dim=1)
    logits = torch.cosine_similarity(trg_anchor.unsqueeze(1),all_feat,dim=2)
    logits /= temperature
    self_logits = torch.cosine_similarity(trg_query.unsqueeze(1), self_feat, dim=2)
    self_logits /= temperature
    labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
    ce_loss = CrossEntropyLoss().cuda()
    return ce_loss(logits,labels),ce_loss(self_logits,labels)

def context_contrast_loss(sample_regions,num_keys=8,temperature=5.,num_pos=3,mine_hard=False,hard_samples=8):
    n, out = sample_regions.shape
    sample_regions = sample_regions.view(n // (num_keys + 1+num_pos), num_keys + 1+num_pos, out)
    anchor, query, keys = sample_regions[:, 0], sample_regions[:, 1:1+num_pos], sample_regions[:, 1+num_pos:]
    loss = 0.
    for i in range(num_pos):
        samples = torch.cat((anchor.unsqueeze(1),query[:,i].unsqueeze(1),keys),dim=1)
        samples = torch.flatten(samples,start_dim=0,end_dim=1)
        loss += contrast_loss(samples,num_keys,temperature,mine_hard=mine_hard,hard_samples=hard_samples)
    loss /= num_pos
    return loss

def contrast_dist_loss(sample_regions,num_keys=8,eps=1.):
    n,out = sample_regions.shape
    sample_regions = sample_regions.view(n//(num_keys+2),num_keys+2,out)
    anchor,query,keys = sample_regions[:,0],sample_regions[:,1],sample_regions[:,2:]
    dist_pos = torch.pow((anchor - query),2)
    # print('before pos sqrt:', torch.mean(dist_pos))
    dist_pos = torch.sqrt(torch.sum(dist_pos,dim=1))/out
    loss = torch.sum(dist_pos)
    dist_neg = torch.pow((anchor.unsqueeze(1)-keys),2)
    # dist_neg = dist_neg*dist_neg
    # print('before sqrt:',torch.mean(dist_neg))
    dist_neg = torch.sqrt(torch.sum(dist_neg,dim=2))/out
    # print('torch.mean',torch.mean(dist_neg),torch.max(dist_neg),torch.min(dist_neg))
    # mask = torch.zeros(dist_neg.shape)
    mask = dist_neg < eps
    len_neg = torch.sum(mask)
    # print('len_neg:',len_neg)
    neg_loss = eps - dist_neg
    neg_loss = torch.sum(neg_loss * mask)
    # neg_loss /= len_neg
    loss += neg_loss
    return loss

def LMI_contrast_loss(trg_embed,src_embed,src_regions,trg_regions,num_keys=100,temperature=0.1):
    '''

    :param trg_embed: n,dim
    :param src_embed: n,dim
    :param src_regions: n*(key+3),dim
    :param trg_regions: n*(key+3),dim
    :param num_keys:
    :param temperature:
    :return:
    '''
    n, out = src_regions.shape
    src_regions = src_regions.view(n // (num_keys + 3), num_keys + 3, out)
    trg_regions = trg_regions.view(n // (num_keys + 3), num_keys + 3, out)
    # src_anchor, trg_query, self_pos, trg_keys = src_regions[:, 0], src_regions[:, 1], src_regions[:, 2], src_regions[:, 3:]  # b,k,c
    # trg_anchor, src_query, src_keys = trg_regions[:, 0], trg_regions[:, 1], trg_regions[:, 3:]
    trg_l = src_regions[:,1:]#n,(2+key),dim
    src_l = trg_regions[:,1:]
    src_lmi_loss,trg_lmi_loss = in_batch_g2l_loss(src_l,src_embed,temperature),in_batch_g2l_loss(trg_l,trg_embed,temperature)
    return src_lmi_loss,trg_lmi_loss

def LMI_cross_contrast_loss(trg_embed,src_embed,src_regions,trg_regions,num_keys=100,temperature=0.1):
    '''

    :param trg_embed: n,dim
    :param src_embed: n,dim
    :param src_regions: n*(key+3),dim
    :param trg_regions: n*(key+3),dim
    :param num_keys:
    :param temperature:
    :return:
    '''
    n, out = src_regions.shape
    src_regions = src_regions.view(n // (num_keys + 3), num_keys + 3, out)
    trg_regions = trg_regions.view(n // (num_keys + 3), num_keys + 3, out)
    # src_anchor, trg_query, self_pos, trg_keys = src_regions[:, 0], src_regions[:, 1], src_regions[:, 2], src_regions[:, 3:]  # b,k,c
    # trg_anchor, src_query, src_keys = trg_regions[:, 0], trg_regions[:, 1], trg_regions[:, 3:]
    trg_l = src_regions[:,1:]#n,(2+key),dim
    src_l = trg_regions[:,1:]
    l_patches = torch.cat((src_l,trg_l),dim=0)
    global_embeddings = torch.cat((src_embed,trg_embed),dim=0)
    lmi_loss = in_batch_g2l_loss(l_patches,global_embeddings,temperature)
    # src_lmi_loss,trg_lmi_loss = in_batch_g2l_loss(src_l,src_embed,temperature),in_batch_g2l_loss(trg_l,trg_embed,temperature)
    return lmi_loss


def mixed_LMI_contrast_loss(trg_embed,src_embed,src_regions,trg_regions,num_keys=100,temperature=0.1):
    '''

    :param trg_embed: n,dim
    :param src_embed: n,dim
    :param src_regions: n*(key+3),dim
    :param trg_regions: n*(key+3),dim
    :param num_keys:
    :param temperature:
    :return:
    '''
    embed = torch.cat([src_embed,trg_embed],dim=0)
    n, out = src_regions.shape
    src_regions = src_regions.view(n // (num_keys + 3), num_keys + 3, out)
    trg_regions = trg_regions.view(n // (num_keys + 3), num_keys + 3, out)
    # src_anchor, trg_query, self_pos, trg_keys = src_regions[:, 0], src_regions[:, 1], src_regions[:, 2], src_regions[:, 3:]  # b,k,c
    # trg_anchor, src_query, src_keys = trg_regions[:, 0], trg_regions[:, 1], trg_regions[:, 3:]
    trg_l = src_regions[:,1:]#n,(2+key),dim
    src_l = trg_regions[:,1:]
    l_patches = torch.cat([src_l,trg_l],dim=0)
    lmi_loss = in_batch_g2l_loss(l_patches,embed,temperature)
    # src_lmi_loss,trg_lmi_loss = in_batch_g2l_loss(src_l,src_embed,trg_l,temperature),in_batch_g2l_loss(trg_l,trg_embed,temperature)
    return lmi_loss

# jinyu: in-batch g2l loss
def in_batch_g2l_loss(l, m, temp=0.1):
    '''
    :param l: local patches
    :param m: global embedding
    :param temp: temperature
    :return:
    '''
    m = m.unsqueeze(1)
    N, n_locals, dim = l.size()
    l_n = l.reshape(-1, dim)  # (N * n_locals) * d
    m_n = m.reshape(-1, dim)  # N * d

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
    u_p = torch.matmul(l, m.permute(0, 2, 1)).unsqueeze(2) / temp  # N * n_locals * 1 * 1

    u_n = torch.mm(m_n, l_n.t()) / temp
    u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1)  # N x N x n_locals x 1

    # We need to mask the diagonal part of the negative tensor.
    mask = torch.eye(N)[:, :, None, None].to(l.device)  # N*N*1*1
    n_mask = 1 - mask
    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)
    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)

    loss = -pred_log[:, :, 0].mean()

    return loss


