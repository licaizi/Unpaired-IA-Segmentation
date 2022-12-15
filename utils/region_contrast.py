import numpy as np
import random
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from models.Multi_Modal_Seg.UNet import Upsample_3D
import torch

def judge_iou(bbox1,bbox2):
    """
     judge whether two boxes have intersections
     :param bbox1:
     :param bbox2:
     :return:
     """
    x1,y1,z1,w1,h1,d1 = bbox1
    x2,y2,z2,w2,h2,d2 = bbox2
    w = min(x1+w1,x2+w2)-max(x1,x2)
    h = min(y1+h1,y2+h2)-max(y1,y2)
    z = min(z1+d1,z2+d2)-max(z1,z2)
    return w<=0 or h<=0 or z<=0

def adjust_boarders(boraders,size):
    """
    adjust borders
    """
    new_boarders = []
    for borader in boraders:
        l,r = borader[0],borader[1]
        gap = r - l
        if l<0:
            l ,r = 0,gap
        elif r> size:
            l,r = size-gap,size
        new_boarders.append((l,r))
    return new_boarders

def sample_positive_tuple(img,gt,size=16,num_pos=3):
    """
    sample multiple boarders for positive regions
    :param img: 
    :param gt: 
    :param size: 
    :param num_pos: 
    :return: sampled boarders of positive regions
    """
    c, W, H, Z = img.shape
    args = np.argwhere(gt.cpu().numpy() == 1)
    min_x, max_x, min_y, max_y, min_z, max_z = min(args[:, 0]), max(args[:, 0]), min(args[:, 1]), max(args[:, 1]), min(
        args[:, 2]), max(args[:, 2])
    center_x, center_y, center_z = int((max_x + min_x) / 2), int((max_y + min_y) / 2), int((max_z + min_z) / 2)
    boarders = list(map(lambda x: (x - size // 2, x + size // 2), [center_x, center_y, center_z]))
    boarders = adjust_boarders(boarders, W)
    pos_anchors = []
    anchor_border = [boarders[0][0], boarders[1][0], boarders[2][0], size, size, size]
    pos_anchors.append(anchor_border)
    for i in range(1,num_pos):
        query_borders = list(map(lambda x: np.random.randint(x - size // 2, x), [center_x, center_y, center_z]))
        query_borders = [(x, x + size) for x in query_borders]
        query_borders = adjust_boarders(query_borders, W)
        query_border = [query_borders[0][0], query_borders[1][0], query_borders[2][0], size, size, size]
        pos_anchors.append(query_border)
    return pos_anchors

def sample_region(img,gt,size=16,num_keys=8):
    """
    Args:
        img:
        gt:
        size:
        num_keys:

    Returns:

    """
    c,W,H,Z = img.shape
    args = np.argwhere(gt.cpu().numpy() == 1)
    # print(args.shape[0],args.shape,'size of nonzero')
    min_x, max_x, min_y, max_y, min_z, max_z = min(args[:, 0]), max(args[:, 0]), min(args[:, 1]), max(args[:, 1]), min(args[:, 2]), max(args[:, 2])
    center_x, center_y, center_z = int((max_x + min_x) / 2), int((max_y + min_y) / 2), int((max_z + min_z) / 2)
    boarders = list(map(lambda x: (x - size // 2, x + size // 2), [center_x, center_y, center_z]))
    boarders = adjust_boarders(boarders, W)
    anchor_border = [boarders[0][0],boarders[1][0],boarders[2][0],size,size,size]

    query_borders = list(map(lambda x: np.random.randint(x-size//2,x), [center_x, center_y, center_z]))
    query_borders = [(x,x+size) for x in query_borders]
    query_borders = adjust_boarders(query_borders,W)
    query_border = [query_borders[0][0], query_borders[1][0], query_borders[2][0], size, size, size]
    count = 0
    sampled_borders = []
    while count < num_keys:
        cx = np.random.randint(0,W-size)
        cy = np.random.randint(0,H-size)
        cz = np.random.randint(0,Z-size)

        sampled_border = [cx,cy,cz,size,size,size]
        if judge_iou(sampled_border,anchor_border):
            # print(cx,cy,cz)
            sampled_borders.append(sampled_border)
            count += 1
    key_regions = [img[:,border[0]:border[0]+size,border[1]:border[1]+size,border[2]:border[2]+size]
                   for border in sampled_borders]
    key_regions = torch.stack(key_regions,dim=0)#k,c,h,w,z
    anchor_regions = img[:, anchor_border[0]:anchor_border[0] + size, anchor_border[1]:anchor_border[1] + size,
                anchor_border[2]:anchor_border[2] + size].unsqueeze(0)
    q_regions = img[:,query_border[0]:query_border[0]+size,query_border[1]:query_border[1]+size,
                query_border[2]:query_border[2]+size].unsqueeze(0)
    return torch.cat((anchor_regions,q_regions,key_regions),dim=0)#(k+1),c,h,w,z

def sample_anchor(img,gt,size=16):
    c, W, H, Z = img.shape
    args = np.argwhere(gt.cpu().numpy() == 1)
    # print(args.shape[0],args.shape,'size of nonzero')
    min_x, max_x, min_y, max_y, min_z, max_z = min(args[:, 0]), max(args[:, 0]), min(args[:, 1]), max(args[:, 1]), min(
        args[:, 2]), max(args[:, 2])
    center_x, center_y, center_z = int((max_x + min_x) / 2), int((max_y + min_y) / 2), int((max_z + min_z) / 2)
    boarders = list(map(lambda x: (x - size // 2, x + size // 2), [center_x, center_y, center_z]))
    boarders = adjust_boarders(boarders, W)
    anchor_border = [boarders[0][0], boarders[1][0], boarders[2][0], size, size, size]
    return anchor_border

def sample_multi_mod_regions(src_img,src_gt,trg_img,trg_gt,size=16,num_keys=8):
    c, W, H, Z = src_img.shape
    src_args = np.argwhere(src_gt.cpu().numpy() == 1)
    min_x, max_x, min_y, max_y, min_z, max_z = min(src_args[:, 0]), max(src_args[:, 0]), min(src_args[:, 1]), max(src_args[:, 1]), min(
        src_args[:, 2]), max(src_args[:, 2])
    center_x, center_y, center_z = int((max_x + min_x) / 2), int((max_y + min_y) / 2), int((max_z + min_z) / 2)

    # print(args.shape[0],args.shape,'size of nonzero')
    src_query_border = sample_anchor(src_img,src_gt,size=size)
    trg_anchor_border = sample_anchor(trg_img,trg_gt,size=size)
    src_count,trg_count = 0,0
    trg_sampled_borders,src_sampled_borders = [],[]
    while src_count < num_keys or trg_count < num_keys:
        cx = np.random.randint(0, W - size)
        cy = np.random.randint(0, H - size)
        cz = np.random.randint(0, Z - size)
        sampled_border = [cx, cy, cz, size, size, size]
        if judge_iou(sampled_border, trg_anchor_border) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
        if judge_iou(sampled_border, src_query_border) and src_count < num_keys:
            # print(cx,cy,cz)
            src_sampled_borders.append(sampled_border)
            src_count += 1
    src_key_regions = [src_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                   for border in src_sampled_borders]
    src_key_regions = torch.stack(src_key_regions, dim=0)  # k,c,h,w,z
    trg_key_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    anchor_regions = trg_img[:, trg_anchor_border[0]:trg_anchor_border[0] + size, trg_anchor_border[1]:trg_anchor_border[1] + size,
                     trg_anchor_border[2]:trg_anchor_border[2] + size].unsqueeze(0)
    q_regions = src_img[:, src_query_border[0]:src_query_border[0] + size, src_query_border[1]:src_query_border[1] + size,
                src_query_border[2]:src_query_border[2] + size].unsqueeze(0)
    return torch.cat((anchor_regions, q_regions, trg_key_regions), dim=0),torch.cat((q_regions, anchor_regions, src_key_regions), dim=0)

def sample_multi_mod_allother_regions(src_img,src_gt,trg_img,trg_gt,size=16,num_keys=8):
    c, W, H, Z = src_img.shape
    src_args = np.argwhere(src_gt.cpu().numpy() == 1)
    src_query_border = sample_anchor(src_img,src_gt,size=size)
    trg_anchor_border = sample_anchor(trg_img,trg_gt,size=size)
    src_count,trg_count = 0,0
    trg_sampled_borders,src_sampled_borders = [],[]
    while src_count < num_keys or trg_count < num_keys:
        cx = np.random.randint(0, W - size)
        cy = np.random.randint(0, H - size)
        cz = np.random.randint(0, Z - size)
        sampled_border = [cx, cy, cz, size, size, size]
        if judge_iou(sampled_border, trg_anchor_border) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
        if judge_iou(sampled_border, src_query_border) and src_count < num_keys:
            # print(cx,cy,cz)
            src_sampled_borders.append(sampled_border)
            src_count += 1
    src_key_regions = [src_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                   for border in src_sampled_borders]
    src_key_regions = torch.stack(src_key_regions, dim=0)  # k,c,h,w,z
    trg_key_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    anchor_regions = trg_img[:, trg_anchor_border[0]:trg_anchor_border[0] + size, trg_anchor_border[1]:trg_anchor_border[1] + size,
                     trg_anchor_border[2]:trg_anchor_border[2] + size].unsqueeze(0)
    q_regions = src_img[:, src_query_border[0]:src_query_border[0] + size, src_query_border[1]:src_query_border[1] + size,
                src_query_border[2]:src_query_border[2] + size].unsqueeze(0)
    return torch.cat((anchor_regions, q_regions, src_key_regions), dim=0),torch.cat((q_regions, anchor_regions, trg_key_regions), dim=0)

def sample_multi_mod_self_other_regions(src_img,src_gt,trg_img,trg_gt,size=16,num_keys=8):
    c, W, H, Z = src_img.shape
    src_args = np.argwhere(src_gt.cpu().numpy() == 1)
    src_query_border = sample_anchor(src_img,src_gt,size=size)
    trg_anchor_border = sample_anchor(trg_img,trg_gt,size=size)
    src_count,trg_count = 0,0
    trg_sampled_borders,src_sampled_borders = [],[]
    while src_count < num_keys or trg_count < num_keys:
        cx = np.random.randint(0, W - size)
        cy = np.random.randint(0, H - size)
        cz = np.random.randint(0, Z - size)
        sampled_border = [cx, cy, cz, size, size, size]
        if judge_iou(sampled_border, trg_anchor_border) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
        if judge_iou(sampled_border, src_query_border) and src_count < num_keys:
            # print(cx,cy,cz)
            src_sampled_borders.append(sampled_border)
            src_count += 1
    src_key_regions = [src_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                   for border in src_sampled_borders]
    src_key_regions = torch.stack(src_key_regions, dim=0)  # k,c,h,w,z
    trg_key_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    anchor_regions = trg_img[:, trg_anchor_border[0]:trg_anchor_border[0] + size, trg_anchor_border[1]:trg_anchor_border[1] + size,
                     trg_anchor_border[2]:trg_anchor_border[2] + size].unsqueeze(0)
    q_regions = src_img[:, src_query_border[0]:src_query_border[0] + size, src_query_border[1]:src_query_border[1] + size,
                src_query_border[2]:src_query_border[2] + size].unsqueeze(0)
    return torch.cat((anchor_regions, q_regions, src_key_regions), dim=0),torch.cat((q_regions, anchor_regions, trg_key_regions), dim=0)

def sample_multi_mod_self_regions(src_img,src_gt,trg_img,trg_gt,size=16,num_keys=8):
    c, W, H, Z = src_img.shape
    src_args = np.argwhere(src_gt.cpu().numpy() == 1)
    src_query_border = sample_anchor(src_img,src_gt,size=size)
    trg_anchor_border = sample_anchor(trg_img,trg_gt,size=size)
    src_count,trg_count = 0,0
    trg_sampled_borders,src_sampled_borders = [],[]
    while src_count < num_keys or trg_count < num_keys:
        cx = np.random.randint(0, W - size)
        cy = np.random.randint(0, H - size)
        cz = np.random.randint(0, Z - size)
        sampled_border = [cx, cy, cz, size, size, size]
        if judge_iou(sampled_border, trg_anchor_border) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
        if judge_iou(sampled_border, src_query_border) and src_count < num_keys:
            # print(cx,cy,cz)
            src_sampled_borders.append(sampled_border)
            src_count += 1
    src_key_regions = [src_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                   for border in src_sampled_borders]
    src_key_regions = torch.stack(src_key_regions, dim=0)  # k,c,h,w,z
    trg_key_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in src_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    anchor_regions = trg_img[:, trg_anchor_border[0]:trg_anchor_border[0] + size, trg_anchor_border[1]:trg_anchor_border[1] + size,
                     trg_anchor_border[2]:trg_anchor_border[2] + size].unsqueeze(0)
    q_regions = src_img[:, src_query_border[0]:src_query_border[0] + size, src_query_border[1]:src_query_border[1] + size,
                src_query_border[2]:src_query_border[2] + size].unsqueeze(0)
    return torch.cat((q_regions, src_key_regions), dim=0),torch.cat((anchor_regions, trg_key_regions), dim=0)

def sample_multi_mod_self_difregions(src_img,src_gt,trg_img,trg_gt,src_size=16,trg_size=16,num_keys=8):
    c, W, H, Z = src_img.shape
    src_args = np.argwhere(src_gt.cpu().numpy() == 1)
    src_query_border = sample_anchor(src_img,src_gt,size=src_size)
    trg_anchor_border = sample_anchor(trg_img,trg_gt,size=trg_size)
    src_count,trg_count = 0,0
    trg_sampled_borders,src_sampled_borders = [],[]
    while src_count < num_keys or trg_count < num_keys:
        cx = np.random.randint(0, W - trg_size)
        cy = np.random.randint(0, H - trg_size)
        cz = np.random.randint(0, Z - trg_size)
        sampled_border = [cx, cy, cz, trg_size, trg_size, trg_size]
        if judge_iou(sampled_border, trg_anchor_border) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
        cx = np.random.randint(0, W - src_size)
        cy = np.random.randint(0, H - src_size)
        cz = np.random.randint(0, Z - src_size)
        sampled_border = [cx, cy, cz, src_size, src_size, src_size]
        if judge_iou(sampled_border, src_query_border) and src_count < num_keys:
            # print(cx,cy,cz)
            src_sampled_borders.append(sampled_border)
            src_count += 1
    src_key_regions = [src_img[:, border[0]:border[0] + src_size, border[1]:border[1] + src_size, border[2]:border[2] + src_size]
                   for border in src_sampled_borders]
    src_key_regions = torch.stack(src_key_regions, dim=0)  # k,c,h,w,z
    trg_key_regions = [trg_img[:, border[0]:border[0] + trg_size, border[1]:border[1] + trg_size, border[2]:border[2] + trg_size]
                       for border in src_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    anchor_regions = trg_img[:, trg_anchor_border[0]:trg_anchor_border[0] + trg_size, trg_anchor_border[1]:trg_anchor_border[1] + trg_size,
                     trg_anchor_border[2]:trg_anchor_border[2] + trg_size].unsqueeze(0)
    q_regions = src_img[:, src_query_border[0]:src_query_border[0] + src_size, src_query_border[1]:src_query_border[1] + src_size,
                src_query_border[2]:src_query_border[2] + src_size].unsqueeze(0)
    return torch.cat((q_regions, src_key_regions), dim=0),torch.cat((anchor_regions, trg_key_regions), dim=0)

def sample_multi_size_positive_tuple(img,gt,sizes=[8,16,32]):
    c, W, H, Z = img.shape
    args = np.argwhere(gt.cpu().numpy() == 1)
    min_x, max_x, min_y, max_y, min_z, max_z = min(args[:, 0]), max(args[:, 0]), min(args[:, 1]), max(args[:, 1]), min(
        args[:, 2]), max(args[:, 2])
    center_x, center_y, center_z = int((max_x + min_x) / 2), int((max_y + min_y) / 2), int((max_z + min_z) / 2)
    s_size,m_size,l_size = sizes[0],sizes[1],sizes[2]
    boarders = list(map(lambda x: (x - s_size // 2, x + s_size // 2), [center_x, center_y, center_z]))
    boarders = adjust_boarders(boarders, W)
    pos_anchors = []
    s_anchor_border = [boarders[0][0], boarders[1][0], boarders[2][0], s_size, s_size, s_size]
    pos_anchors.append(s_anchor_border)
    m_boarders = list(map(lambda x: (x - m_size // 2, x + m_size // 2), [center_x, center_y, center_z]))
    boarders = adjust_boarders(m_boarders, W)
    m_anchor_border = [boarders[0][0], boarders[1][0], boarders[2][0], m_size, m_size, m_size]
    pos_anchors.append(m_anchor_border)
    l_boarders = list(map(lambda x: (x - l_size // 2, x + l_size // 2), [center_x, center_y, center_z]))
    boarders = adjust_boarders(l_boarders, W)
    l_anchor_border = [boarders[0][0], boarders[1][0], boarders[2][0], l_size, l_size, l_size]
    pos_anchors.append(l_anchor_border)

    return pos_anchors

def sample_multi_modal_size_regions(src_img,src_gt,trg_img,trg_gt,num_keys=8,pos_num=3,sizes=[8,16,32]):
    '''
    generate multi positive samples and mine hard negative samples
    :param src_img:
    :param src_gt:
    :param trg_img:
    :param trg_gt:
    :param sizes:
    :param num_keys:
    :return:
    '''
    c, W, H, Z = src_img.shape
    size = sizes[1]
    src_query_borders = sample_multi_size_positive_tuple(src_img, src_gt,sizes=sizes)
    trg_query_borders = sample_multi_size_positive_tuple(trg_img, trg_gt,sizes=sizes)
    trg_anchor_border = trg_query_borders[1]
    src_anchor_border = src_query_borders[1]
    src_count,trg_count = 0,0
    trg_sampled_borders,src_sampled_borders = [],[]
    up_module = Upsample_3D(size=sizes[1],mode='trilinear')
    down_module = Upsample_3D(size=sizes[1], mode='trilinear')
    while src_count < num_keys or trg_count < num_keys:
        cx = np.random.randint(0, W - size)
        cy = np.random.randint(0, H - size)
        cz = np.random.randint(0, Z - size)
        sampled_border = [cx, cy, cz, size, size, size]
        if judge_iou(sampled_border, trg_query_borders[1]) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
        if judge_iou(sampled_border, src_query_borders[1]) and src_count < num_keys:
            # print(cx,cy,cz)
            src_sampled_borders.append(sampled_border)
            src_count += 1
    src_key_regions = [src_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                   for border in src_sampled_borders]
    src_key_regions = torch.stack(src_key_regions, dim=0)  # k,c,h,w,z
    trg_key_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    trg_anchor_region = trg_img[:, trg_anchor_border[0]:trg_anchor_border[0] + size, trg_anchor_border[1]:trg_anchor_border[1] + size,
                     trg_anchor_border[2]:trg_anchor_border[2] + size].unsqueeze(0)
    src_anchor_region = src_img[:, src_anchor_border[0]:src_anchor_border[0] + size, src_anchor_border[1]:src_anchor_border[1] + size,
                src_anchor_border[2]:src_anchor_border[2] + size].unsqueeze(0)
    src_pos_regions = [src_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                   for border in src_query_borders]
    src_pos_regions[0] = up_module(src_pos_regions[0].unsqueeze(0)).squeeze(0)
    src_pos_regions[2] = down_module(src_pos_regions[2].unsqueeze(0)).squeeze(0)
    src_pos_regions = torch.stack(src_pos_regions,dim=0)
    trg_pos_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_query_borders]
    trg_pos_regions[0] = up_module(trg_pos_regions[0].unsqueeze(0)).squeeze(0)
    trg_pos_regions[2] = down_module(trg_pos_regions[2].unsqueeze(0)).squeeze(0)
    trg_pos_regions = torch.stack(trg_pos_regions,dim=0)
    return torch.cat((trg_anchor_region, src_pos_regions, src_key_regions), dim=0),\
           torch.cat((src_anchor_region, trg_pos_regions, trg_key_regions), dim=0)

def sample_multi_mod_self_context_difregions(src_img,src_gt,trg_img,trg_gt,src_size=16,trg_size=16,num_keys=8,pos_num=2):
    c, W, H, Z = src_img.shape
    src_args = np.argwhere(src_gt.cpu().numpy() == 1)
    src_query_borders = sample_positive_tuple(src_img, src_gt, size=src_size, num_pos=pos_num)
    trg_query_borders = sample_positive_tuple(trg_img, trg_gt, size=trg_size, num_pos=pos_num)
    trg_anchor_border = trg_query_borders[0]
    src_anchor_border = src_query_borders[0]
    src_count,trg_count = 0,0
    trg_sampled_borders,src_sampled_borders = [],[]
    while src_count < num_keys or trg_count < num_keys:
        cx = np.random.randint(0, W - trg_size)
        cy = np.random.randint(0, H - trg_size)
        cz = np.random.randint(0, Z - trg_size)
        sampled_border = [cx, cy, cz, trg_size, trg_size, trg_size]
        if judge_iou(sampled_border, trg_anchor_border) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
        cx = np.random.randint(0, W - src_size)
        cy = np.random.randint(0, H - src_size)
        cz = np.random.randint(0, Z - src_size)
        sampled_border = [cx, cy, cz, src_size, src_size, src_size]
        if judge_iou(sampled_border, src_anchor_border) and src_count < num_keys:
            # print(cx,cy,cz)
            src_sampled_borders.append(sampled_border)
            src_count += 1
    src_key_regions = [src_img[:, border[0]:border[0] + src_size, border[1]:border[1] + src_size, border[2]:border[2] + src_size]
                       for border in src_sampled_borders]
    src_key_regions = torch.stack(src_key_regions, dim=0)  # k,c,h,w,z
    trg_key_regions = [trg_img[:, border[0]:border[0] + trg_size, border[1]:border[1] + trg_size, border[2]:border[2] + trg_size]
                       for border in src_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    src_pos_regions = [src_img[:, border[0]:border[0] + src_size, border[1]:border[1] + src_size, border[2]:border[2] + src_size]
                       for border in src_query_borders]
    src_pos_regions = torch.stack(src_pos_regions, dim=0)
    trg_pos_regions = [trg_img[:, border[0]:border[0] + trg_size, border[1]:border[1] + trg_size, border[2]:border[2] + trg_size]
                       for border in trg_query_borders]
    trg_pos_regions = torch.stack(trg_pos_regions, dim=0)
    return torch.cat((trg_pos_regions, trg_key_regions), dim=0),\
           torch.cat((src_pos_regions, src_key_regions), dim=0)
def sample_onemodal_regions(trg_img,trg_gt,size=16,num_keys=8,num_pos=2):
    c, W, H, Z = trg_img.shape
    # src_args = np.argwhere(src_gt.cpu().numpy() == 1)
    # src_query_border = sample_anchor(src_img, src_gt, size=size)
    trg_anchor_borders = sample_positive_tuple(trg_img, trg_gt, size=size,num_pos=num_pos)
    trg_anchor_border = trg_anchor_borders[0]
    src_count, trg_count = 0, 0
    trg_sampled_borders = []
    while trg_count < num_keys:
        cx = np.random.randint(0, W - size)
        cy = np.random.randint(0, H - size)
        cz = np.random.randint(0, Z - size)
        sampled_border = [cx, cy, cz, size, size, size]
        if judge_iou(sampled_border, trg_anchor_border) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
    trg_key_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    trg_pos_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_anchor_borders]
    trg_pos_regions = torch.stack(trg_pos_regions, dim=0)
    return torch.cat((trg_pos_regions, trg_key_regions), dim=0)

def sample_multi_modal_context_regions(src_img,src_gt,trg_img,trg_gt,size=16,num_keys=8,pos_num=3):
    '''
    generate multi positive samples and mine hard negative samples
    :param src_img:
    :param src_gt:
    :param trg_img:
    :param trg_gt:
    :param size:
    :param num_keys:
    :return:src_sample_regions contain trg anchor,src positive regions and src negative regions,
    trg_sample_regions contain src anchor,trg positive regions and trg negative regions
    '''
    c, W, H, Z = src_img.shape
    src_query_borders = sample_positive_tuple(src_img, src_gt, size=size,num_pos=pos_num)
    trg_query_borders = sample_positive_tuple(trg_img, trg_gt, size=size,num_pos=pos_num)
    trg_anchor_border = trg_query_borders[0]
    src_anchor_border = src_query_borders[0]
    src_count,trg_count = 0,0
    trg_sampled_borders,src_sampled_borders = [],[]
    while src_count < num_keys or trg_count < num_keys:
        cx = np.random.randint(0, W - size)
        cy = np.random.randint(0, H - size)
        cz = np.random.randint(0, Z - size)
        sampled_border = [cx, cy, cz, size, size, size]
        if judge_iou(sampled_border, trg_query_borders[0]) and trg_count < num_keys:
            # print(cx,cy,cz)
            trg_sampled_borders.append(sampled_border)
            trg_count += 1
        if judge_iou(sampled_border, src_query_borders[0]) and src_count < num_keys:
            # print(cx,cy,cz)
            src_sampled_borders.append(sampled_border)
            src_count += 1
    src_key_regions = [src_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                   for border in src_sampled_borders]
    src_key_regions = torch.stack(src_key_regions, dim=0)  # k,c,h,w,z
    trg_key_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_sampled_borders]
    trg_key_regions = torch.stack(trg_key_regions, dim=0)
    trg_anchor_region = trg_img[:, trg_anchor_border[0]:trg_anchor_border[0] + size, trg_anchor_border[1]:trg_anchor_border[1] + size,
                     trg_anchor_border[2]:trg_anchor_border[2] + size].unsqueeze(0)
    src_anchor_region = src_img[:, src_anchor_border[0]:src_anchor_border[0] + size, src_anchor_border[1]:src_anchor_border[1] + size,
                src_anchor_border[2]:src_anchor_border[2] + size].unsqueeze(0)
    src_pos_regions = [src_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                   for border in src_query_borders]
    src_pos_regions = torch.stack(src_pos_regions,dim=0)
    trg_pos_regions = [trg_img[:, border[0]:border[0] + size, border[1]:border[1] + size, border[2]:border[2] + size]
                       for border in trg_query_borders]
    trg_pos_regions = torch.stack(trg_pos_regions,dim=0)
    return torch.cat((trg_anchor_region, src_pos_regions, src_key_regions), dim=0),\
           torch.cat((src_anchor_region, trg_pos_regions, trg_key_regions), dim=0)


def mine_hard_negsample(anchor,keys,hard_num=100):
    sim = torch.cosine_similarity(anchor.unsqueeze(1), keys, dim=2)#b,k
    result,kmax_index = torch.topk(sim.unsqueeze(1),k=hard_num,dim=2)
    hard_neg = [keys[i,kmax_index[i,0,:]] for i in range(keys.shape[0])]
    hard_neg = torch.stack(hard_neg,dim=0)#b,k,c
    return hard_neg

def sample_hard_neg(anchor,keys,hard_num=100,temperature=0.1):
    sim = torch.cosine_similarity(anchor.unsqueeze(1).cpu(), keys.cpu(), dim=2).type(torch.float32)  # b,k
    b,k = sim.shape
    # print('shape:',sim.shape,hard_num)
    sim = torch.softmax((sim/temperature).cpu(),dim=1)
    # sim = torch.clamp(sim,0,1).cpu()
    # non_zeros0,non_zeros1 = len(torch.where(sim[0] > 0)[0]),len(torch.where(sim[1] > 0)[0])
    # if non_zeros0 < hard_num or non_zeros1 < hard_num:
    #     print('not enough hard samples')
    #     return keys
    indexs = torch.multinomial(sim,num_samples=hard_num)
    indexs = indexs.cuda()
    hard_negs = [keys[i,indexs[i,:]] for i in range(b)]
    # hard_negs = [torch.index_select(keys,0,index=indexs[i]) for i in range(b)]
    hard_negs = torch.stack(hard_negs,dim=0)
    return hard_negs


def is_no_target(train_batch):
    train_label = train_batch['target']
    b,c,h,w,z = train_label.shape
    for i in range(b):
        label = train_label[i]
        if len(torch.unique(label)) == 1:
            return True
    return False

def generate_batch_mulmod_regions(src_img_batch,src_gt_batch,trg_img_batch,trg_gt_batch,size=16,num_keys=8,moco=False):
    bth_size = src_img_batch.shape[0]
    # anchor_regions = []
    src_sampled_regions,trg_sampled_regions = [],[]
    for i in range(bth_size):
        trg_sampled_region,src_sampled_region = sample_multi_mod_regions(src_img_batch[i,:],src_gt_batch[i,0],trg_img_batch[i,:],trg_gt_batch[i,0],size,num_keys)
        src_sampled_regions.append(src_sampled_region)
        trg_sampled_regions.append(trg_sampled_region)
        # anchor_regions.append(anchor_region)
    src_sampled_regions = torch.stack(src_sampled_regions,dim=0)#b,(2+k),c,h,w,z
    b,k2,c,h,w,z = src_sampled_regions.shape
    trg_sampled_regions = torch.stack(trg_sampled_regions, dim=0)  # b,(2+k),c,h,w,z
    if moco:
        return src_sampled_regions,trg_sampled_regions
    src_sampled_regions = src_sampled_regions.view(b * k2, c, h, w, z)
    trg_sampled_regions = trg_sampled_regions.view(b * k2, c, h, w, z)
    return src_sampled_regions,trg_sampled_regions

def generate_batch_onemod_regions(trg_img_batch,trg_gt_batch,num_pos=3,
                                          size=16,num_keys=8,):
    bth_size = trg_img_batch.shape[0]
    # anchor_regions = []
    trg_sampled_regions = []
    for i in range(bth_size):
        trg_sampled_region = sample_onemodal_regions(trg_img_batch[i, :],trg_gt_batch[i, 0], size, num_keys, num_pos=num_pos)
        # src_sampled_regions.append(src_sampled_region)
        trg_sampled_regions.append(trg_sampled_region)
    trg_sampled_regions = torch.stack(trg_sampled_regions, dim=0)  # b,(1+3+k),c,h,w,z
    b, k2, c, h, w, z = trg_sampled_regions.shape
    trg_sampled_regions = trg_sampled_regions.view(b * k2, c, h, w, z)
    return trg_sampled_regions


def generate_batch_eachmod_regions(src_img_batch,src_gt_batch,trg_img_batch,trg_gt_batch,size=16,num_keys=8,moco=False):
    bth_size = src_img_batch.shape[0]
    # anchor_regions = []
    src_sampled_regions,trg_sampled_regions = [],[]
    for i in range(bth_size):
        src_sampled_region,trg_sampled_region = sample_multi_mod_self_regions(src_img_batch[i,:],src_gt_batch[i,0],trg_img_batch[i,:],trg_gt_batch[i,0],size,num_keys)
        src_sampled_regions.append(src_sampled_region)
        trg_sampled_regions.append(trg_sampled_region)
        # anchor_regions.append(anchor_region)
    src_sampled_regions = torch.stack(src_sampled_regions,dim=0)#b,(1+k),c,h,w,z
    b,k2,c,h,w,z = src_sampled_regions.shape
    trg_sampled_regions = torch.stack(trg_sampled_regions, dim=0)  # b,(1+k),c,h,w,z
    if moco:
        return src_sampled_regions,trg_sampled_regions
    src_sampled_regions = src_sampled_regions.view(b * k2, c, h, w, z)
    trg_sampled_regions = trg_sampled_regions.view(b * k2, c, h, w, z)
    return src_sampled_regions,trg_sampled_regions

def generate_batch_eachmod_dif_regions(src_img_batch,src_gt_batch,trg_img_batch,trg_gt_batch,src_size=16,trg_size=16,num_keys=8,moco=False):
    bth_size = src_img_batch.shape[0]
    # anchor_regions = []
    src_sampled_regions,trg_sampled_regions = [],[]
    for i in range(bth_size):
        src_sampled_region,trg_sampled_region = sample_multi_mod_self_difregions(src_img_batch[i,:],src_gt_batch[i,0],trg_img_batch[i,:],
                                                                              trg_gt_batch[i,0],src_size,trg_size,num_keys)
        src_sampled_regions.append(src_sampled_region)
        trg_sampled_regions.append(trg_sampled_region)
        # anchor_regions.append(anchor_region)
    src_sampled_regions = torch.stack(src_sampled_regions,dim=0)#b,(1+k),c,h,w,z
    b,k1,c,h,w,z = src_sampled_regions.shape
    trg_sampled_regions = torch.stack(trg_sampled_regions, dim=0)  # b,(1+k),c,h,w,z
    b, k2, c_, h_, w_, z_ = trg_sampled_regions.shape
    if moco:
        return src_sampled_regions,trg_sampled_regions
    src_sampled_regions = src_sampled_regions.view(b * k1, c, h, w, z)
    trg_sampled_regions = trg_sampled_regions.view(b * k2, c_, h_, w_, z_)
    return src_sampled_regions,trg_sampled_regions

def generate_batch_mulmod_multisize_context_regions(src_img_batch,src_gt_batch,trg_img_batch,trg_gt_batch,num_pos=3,
                                          size=16,num_keys=8,moco=False):
    bth_size = src_img_batch.shape[0]
    # anchor_regions = []
    src_sampled_regions,trg_sampled_regions = [],[]
    for i in range(bth_size):
        trg_sampled_region,src_sampled_region = sample_multi_modal_size_regions(src_img_batch[i,:],src_gt_batch[i,0],trg_img_batch[i,:],
                                                                                   trg_gt_batch[i,0],num_keys,pos_num=num_pos,sizes=[8,16,32])
        src_sampled_regions.append(src_sampled_region)
        trg_sampled_regions.append(trg_sampled_region)
        # anchor_regions.append(anchor_region)
    src_sampled_regions = torch.stack(src_sampled_regions,dim=0)#b,(1+3+k),c,h,w,z
    b,k2,c,h,w,z = src_sampled_regions.shape
    trg_sampled_regions = torch.stack(trg_sampled_regions, dim=0)  # b,(1+3+k),c,h,w,z
    if moco:
        return src_sampled_regions,trg_sampled_regions
    src_sampled_regions = src_sampled_regions.view(b * k2, c, h, w, z)
    trg_sampled_regions = trg_sampled_regions.view(b * k2, c, h, w, z)
    return src_sampled_regions,trg_sampled_regions


def generate_batch_eachmod_dif_context_regions(src_img_batch,src_gt_batch,trg_img_batch,trg_gt_batch,src_size=16,trg_size=16,
                                               num_keys=8,moco=False,num_pos=2):
    bth_size = src_img_batch.shape[0]
    # anchor_regions = []
    src_sampled_regions,trg_sampled_regions = [],[]
    for i in range(bth_size):
        trg_sampled_region,src_sampled_region = sample_multi_mod_self_context_difregions(src_img_batch[i,:],src_gt_batch[i,0],trg_img_batch[i,:],
                                                                              trg_gt_batch[i,0],src_size,trg_size,num_keys,pos_num=2)
        src_sampled_regions.append(src_sampled_region)
        trg_sampled_regions.append(trg_sampled_region)
        # anchor_regions.append(anchor_region)
    src_sampled_regions = torch.stack(src_sampled_regions,dim=0)#b,(2+k),c,h,w,z
    b,k1,c,h,w,z = src_sampled_regions.shape
    trg_sampled_regions = torch.stack(trg_sampled_regions, dim=0)  # b,(2+k),c,h,w,z
    b, k2, c_, h_, w_, z_ = trg_sampled_regions.shape
    if moco:
        return src_sampled_regions,trg_sampled_regions
    src_sampled_regions = src_sampled_regions.view(b * k1, c, h, w, z)
    trg_sampled_regions = trg_sampled_regions.view(b * k2, c_, h_, w_, z_)
    return src_sampled_regions,trg_sampled_regions


def generate_batch_mulmod_allother_regions(src_img_batch,src_gt_batch,trg_img_batch,trg_gt_batch,size=16,num_keys=8,moco=False):
    bth_size = src_img_batch.shape[0]
    # anchor_regions = []
    src_sampled_regions,trg_sampled_regions = [],[]
    for i in range(bth_size):
        trg_sampled_region,src_sampled_region = sample_multi_mod_allother_regions(src_img_batch[i,:],src_gt_batch[i,0],trg_img_batch[i,:],trg_gt_batch[i,0],size,num_keys)
        src_sampled_regions.append(src_sampled_region)
        trg_sampled_regions.append(trg_sampled_region)
        # anchor_regions.append(anchor_region)
    src_sampled_regions = torch.stack(src_sampled_regions,dim=0)#b,(2+k),c,h,w,z
    b,k2,c,h,w,z = src_sampled_regions.shape
    trg_sampled_regions = torch.stack(trg_sampled_regions, dim=0)  # b,(2+k),c,h,w,z
    if moco:
        return src_sampled_regions,trg_sampled_regions
    src_sampled_regions = src_sampled_regions.view(b * k2, c, h, w, z)
    trg_sampled_regions = trg_sampled_regions.view(b * k2, c, h, w, z)
    return src_sampled_regions,trg_sampled_regions

def generate_batch_mulmod_context_regions(src_img_batch,src_gt_batch,trg_img_batch,trg_gt_batch,num_pos=3,
                                          size=16,num_keys=8,moco=False):
    """
    generate positive and negative regions for multi modality
    Args:
        src_img_batch:
        src_gt_batch:
        trg_img_batch:
        trg_gt_batch:
        num_pos:
        size:
        num_keys:
        moco:

    Returns:

    """
    bth_size = src_img_batch.shape[0]
    # anchor_regions = []
    src_sampled_regions,trg_sampled_regions = [],[]
    for i in range(bth_size):
        trg_sampled_region,src_sampled_region = sample_multi_modal_context_regions(src_img_batch[i,:],src_gt_batch[i,0],trg_img_batch[i,:],
                                                                                   trg_gt_batch[i,0],size,num_keys,pos_num=num_pos)
        src_sampled_regions.append(src_sampled_region)
        trg_sampled_regions.append(trg_sampled_region)
        # anchor_regions.append(anchor_region)
    src_sampled_regions = torch.stack(src_sampled_regions,dim=0)#b,(1+3+k),c,h,w,z
    b,k2,c,h,w,z = src_sampled_regions.shape
    trg_sampled_regions = torch.stack(trg_sampled_regions, dim=0)  # b,(1+3+k),c,h,w,z
    if moco:
        return src_sampled_regions,trg_sampled_regions
    src_sampled_regions = src_sampled_regions.view(b * k2, c, h, w, z)
    trg_sampled_regions = trg_sampled_regions.view(b * k2, c, h, w, z)
    return src_sampled_regions,trg_sampled_regions

def generate_batch_regions(img_batch, gt_batch, size=16, num_keys=8):
    bth_size = img_batch.shape[0]
    # anchor_regions = []
    sampled_regions = []
    for i in range(bth_size):
        sampled_region = sample_region(img_batch[i, :], gt_batch[i, 0], size, num_keys)
        sampled_regions.append(sampled_region)
        # anchor_regions.append(anchor_region)
    sampled_regions = torch.stack(sampled_regions, dim=0)  # b,(2+k),c,h,w,z
    b, k2, c, h, w, z = sampled_regions.shape
    sampled_regions = sampled_regions.view(b * k2, c, h, w, z)
    return sampled_regions

