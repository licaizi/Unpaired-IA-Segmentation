import numpy as np

def adjust_border(center,shape,crop_size):
    lt = np.clip(center - crop_size//2,0,shape)
    rt = np.clip(center + crop_size//2,0,shape)
    return lt,rt

def adapt_border(lt,rt,bb1,bb2):
    if rt - lt < bb2 - bb1:
        length = rt - lt
        bb2 = bb1 + length
        # print('adapted:',lt,rt,bb1,bb2)
    elif rt - lt > bb2 - bb1:
        length = bb2 - bb1
        rt = lt + length
        # print('adapted2:', lt, rt, bb1, bb2)
    return lt,rt,bb1,bb2

def cut_bbox(img,gt,lam):
    size = img.shape
    W = size[0]
    H = size[1]
    Z = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cut_z = np.int(Z*cut_rat)
    args = np.argwhere(gt.cpu().numpy() == 1)
    min_x, max_x, min_y, max_y, min_z, max_z = min(args[:, 0]), max(args[:, 0]), min(args[:, 1]), max(args[:, 1]), min(args[:, 2]), max(args[:, 2])
    center_x, center_y, center_z = int((max_x + min_x) / 2), int((max_y + min_y) / 2), int((max_z + min_z) / 2)
    # print(min(args[:, 0]), max(args[:, 0]), min(args[:, 1]), max(args[:, 1]), min(args[:, 2]), max(args[:, 2]))
    # print(center_x, center_y, center_z)
    lt_x, rb_x = adjust_border(center_x, img.shape[0], cut_w)
    lt_y, rb_y = adjust_border(center_y, img.shape[1], cut_h)
    lt_s, rb_s = adjust_border(center_z, img.shape[2], cut_z)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(Z)

    bbx1,bbx2 = adjust_border(cx,W,cut_w)
    bby1,bby2 = adjust_border(cy,H,cut_h)
    bbz1,bbz2 = adjust_border(cz,Z,cut_z)
    # print(rb_x-lt_x,bbx2-bbx1,rb_y-lt_y,bby2-bby1,rb_s-lt_s,bbz2-bbz1)
    # print([bbx1, bby1, bbz1, bbx2, bby2, bbz2, lt_x, rb_x, lt_y, rb_y, lt_s, rb_s])
    lt_x,rb_x,bbx1,bbx2 = adapt_border(lt_x,rb_x,bbx1,bbx2)
    lt_y, rb_y, bby1, bby2 = adapt_border(lt_y, rb_y, bby1, bby2)
    lt_s, rb_s, bbz1, bbz2 = adapt_border(lt_s, rb_s, bbz1, bbz2)
    # print(rb_x - lt_x, bbx2 - bbx1, rb_y - lt_y, bby2 - bby1, rb_s - lt_s, bbz2 - bbz1)
    # print([bbx1, bby1,bbz1, bbx2, bby2,bbz2,lt_x,rb_x,lt_y,rb_y,lt_s,rb_s])
    # bbx1 = np.clip(cx - cut_w // 2, 0, W)
    # bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbz1 = np.clip(cz - cut_z//2,0,Z)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    # bby2 = np.clip(cy + cut_h // 2, 0, H)
    # bbz2 = np.clip(cz + cut_z//2, 0, Z)

    return [bbx1, bby1,bbz1, bbx2, bby2,bbz2,lt_x,rb_x,lt_y,rb_y,lt_s,rb_s]
import copy
import matplotlib.pyplot as plt
def get_mix_batch(src_batch,trg_batch,src_labels,trg_labels,lam):
    btch_size = src_batch.shape[0]
    cutmix_trg_batch = copy.deepcopy(trg_batch)
    cutmix_trg_label = copy.deepcopy(trg_labels)
    bboxs = []
    for i in range(btch_size):
        bbox = cut_bbox(src_batch[i,0],src_labels[i,0],lam)
        [bbx1, bby1, bbz1, bbx2, bby2, bbz2, lt_x, rb_x, lt_y, rb_y, lt_s, rb_s] = bbox
        bboxs.append(bbox)
        # plt.subplot(1,2,1)
        # plt.imshow(src_batch[i,0,lt_x:rb_x,lt_y:rb_y,lt_s:rb_s],cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(src_labels[i, 0, lt_x:rb_x, lt_y:rb_y, lt_s:rb_s], cmap='gray')
        # plt.show()
        cutmix_trg_batch[i,:,bbx1:bbx2,bby1:bby2,bbz1:bbz2] = src_batch[i,:,lt_x:rb_x,lt_y:rb_y,lt_s:rb_s]
        cutmix_trg_label[i,:,bbx1:bbx2,bby1:bby2,bbz1:bbz2] = src_labels[i,:,lt_x:rb_x,lt_y:rb_y,lt_s:rb_s]

    # bboxs = np.stack(bboxs,axis=0)
    return cutmix_trg_batch,cutmix_trg_label,bboxs

def get_mixed_preds(bboxs,trg_out,src_out):
    btch_size = trg_out.shape[0]
    for i in range(btch_size):
        [bbx1, bby1, bbz1, bbx2, bby2, bbz2, lt_x, rb_x, lt_y, rb_y, lt_s, rb_s] = bboxs[i]
        trg_out[i,:,bbx1:bbx2,bby1:bby2,bbz1:bbz2] = src_out[i,:,lt_x:rb_x,lt_y:rb_y,lt_s:rb_s]
    return trg_out

# for i in range(10):
#     lam = np.random.beta(1, 1)
#     print(lam)

# data = np.array([1,2,3,4])
# data2 = np.array([3,4,5,6])
# data3 = np.stack((data,data2))
# print(data3.shape)