import os
import torch
import shutil
import numpy as np
from models.Model_Utils import softmax_helper
from batchgenerators.augmentations.utils import pad_nd_image
from utils.Metrics import dice,jaccard,hausdorff_distance_95,sensitivity
from data_preprocessing.Data_Utils import convert_to_one_hot,save_nii
import math
import random

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def save_checkpoint(state, is_best, save_path, filename, bestname,save_best_only=False):
    filename = os.path.join(save_path, filename)
    if not save_best_only:
        torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, bestname)
        if save_best_only:
            torch.save(state,bestname)
        else:
            shutil.copyfile(filename, bestname)
def save_final_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def pad_img_to_fit_network(img, num_pool):
    """
    we pad an image to make sure the width and height can be divided exactly by pow(2, num_pool),
    thus we can predict an any size image
    :param img:(b, h, w)
    :return:
    """
    factor = (pow(2, num_pool),) * len(img.shape[-2:])
    padded_img, slicer = pad_nd_image(img, new_shape=None, mode='constant', return_slicer=True,
                                      shape_must_be_divisible_by=factor)
    # (13, 216, 256) (13, 224, 256) [slice(0, 13, None), slice(4, 220, None), slice(0, 256, None)]
    return padded_img, slicer

def generate_img_patches(img, patch_size, ita):  # (c, d, h, w)
    ss_c, ss_h, ss_w, ss_l = img.shape

    # pad the img if the size is smaller than the crop size
    padding_size_x = max(0, math.ceil((patch_size[0] - ss_h) / 2))
    padding_size_y = max(0, math.ceil((patch_size[1] - ss_w) / 2))
    padding_size_z = max(0, math.ceil((patch_size[2] - ss_l) / 2))
    img = np.pad(img, ((0, 0), (padding_size_x, padding_size_x), (padding_size_y, padding_size_y),
                       (padding_size_z, padding_size_z)), 'constant')

    ss_c, ss_h, ss_w, ss_l = img.shape

    fold_h = math.floor(ss_h / patch_size[0]) + ita
    fold_w = math.floor(ss_w / patch_size[1]) + ita
    fold_l = math.floor(ss_l / patch_size[2]) + ita
    overlap_h = int(math.ceil((ss_h - patch_size[0]) / (fold_h - 1)))
    overlap_w = int(math.ceil((ss_w - patch_size[1]) / (fold_w - 1)))
    overlap_l = int(math.ceil((ss_l - patch_size[2]) / (fold_l - 1)))
    idx_h = [0] if overlap_h == 0 else [i for i in range(0, ss_h - patch_size[0] + 1, overlap_h)]
    idx_h.append(ss_h - patch_size[0])
    idx_h = np.unique(idx_h)
    idx_w = [0] if overlap_w == 0 else [i for i in range(0, ss_w - patch_size[1] + 1, overlap_w)]
    idx_w.append(ss_w - patch_size[1])
    idx_w = np.unique(idx_w)
    idx_l = [0] if overlap_l == 0 else [i for i in range(0, ss_l - patch_size[2] + 1, overlap_l)]
    idx_l.append(ss_l - patch_size[2])
    idx_l = np.unique(idx_l)

    crop_data_list = []
    for itr_h in idx_h:
        for itr_w in idx_w:
            for itr_l in idx_l:
                crop_data = img[:, itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                            itr_l: itr_l + patch_size[2]]
                crop_data_list.append(crop_data)
    return crop_data_list, ss_c, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z

def generate_d_map_patch2Img(patch_list, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z,
                             patch_size, ita):
    label_array = np.zeros((ss_h, ss_w, ss_l))
    cnt_array = np.zeros((ss_h, ss_w, ss_l))
    fold_h = math.floor(ss_h / patch_size[0]) + ita
    fold_w = math.floor(ss_w / patch_size[1]) + ita
    fold_l = math.floor(ss_l / patch_size[2]) + ita
    overlap_h = int(math.ceil((ss_h - patch_size[0]) / (fold_h - 1)))
    overlap_w = int(math.ceil((ss_w - patch_size[1]) / (fold_w - 1)))
    overlap_l = int(math.ceil((ss_l - patch_size[2]) / (fold_l - 1)))
    idx_h = [0] if overlap_h == 0 else [i for i in range(0, ss_h - patch_size[0] + 1, overlap_h)]
    idx_h.append(ss_h - patch_size[0])
    idx_h = np.unique(idx_h)
    idx_w = [0] if overlap_w == 0 else [i for i in range(0, ss_w - patch_size[1] + 1, overlap_w)]
    idx_w.append(ss_w - patch_size[1])
    idx_w = np.unique(idx_w)
    idx_l = [0] if overlap_l == 0 else [i for i in range(0, ss_l - patch_size[2] + 1, overlap_l)]
    idx_l.append(ss_l - patch_size[2])
    idx_l = np.unique(idx_l)

    p_count = 0
    for itr_h in idx_h:
        for itr_w in idx_w:
            for itr_l in idx_l:
                label_array[itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                itr_l: itr_l + patch_size[2]] += patch_list[p_count]
                cnt_array[itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                itr_l: itr_l + patch_size[2]] += 1
                p_count += 1

    vote_label = label_array / cnt_array
    score_map = vote_label[padding_size_x: vote_label.shape[0] - padding_size_x,
                padding_size_y: vote_label.shape[1] - padding_size_y,
                padding_size_z: vote_label.shape[2] - padding_size_z]

    return score_map

def validation(model, dataset, all_indice=False,dsbn=False,return_name=False):
    dices = []
    jaccards = []
    sensitivitys = []
    hd_95s = []
    dices_withname = {}
    for key in dataset.keys():
        img_ed = dataset[key]['img']     # (13, 216, 256)
        img_ed_gt = dataset[key]['gt']   # (13, 4, 216, 256)
        patch_size = (128, 128, 128)
        # crop image
        img_ed = np.expand_dims(img_ed, axis=0)
        center_pt = dataset[key]['center'][0]#random.choice(dataset[key]['center'])
        lt_x = int(max(0, center_pt[0] - patch_size[0]/2))
        lt_y = int(max(0, center_pt[1] - patch_size[1]/2))
        lt_s = int(max(0, center_pt[2] - patch_size[2]/2))
        rb_x = int(min(img_ed.shape[2], lt_x + patch_size[0]))
        rb_y = int(min(img_ed.shape[3], lt_y + patch_size[1]))
        rb_s = int(min(img_ed.shape[1], lt_s + patch_size[2]))
        crop_img = np.zeros((1, 128, 128, 128))
        crop_img[:, :rb_s-lt_s, :rb_x-lt_x, :rb_y-lt_y] = img_ed[:, lt_s:rb_s, lt_x:rb_x, lt_y:rb_y]  # in case that crop length < 128
        crop_img_pad = crop_img
        input_data = np.expand_dims(crop_img_pad, axis=0)
        if dsbn:
            crop_output = softmax_helper(model(torch.from_numpy(input_data).float().cuda(),[1])).detach().cpu().numpy()
        else:
            crop_output = softmax_helper(model(torch.from_numpy(input_data).float().cuda())).detach().cpu().numpy()
        # crop_output = model(torch.from_numpy(input_data).float().cuda()).detach().cpu().numpy()
        patch_pred = crop_output.squeeze().argmax(0)
        pred_map = np.zeros(img_ed.shape[1:], dtype=np.int64)
        pred_map[lt_s:rb_s, lt_x:rb_x, lt_y:rb_y] = patch_pred[:rb_s-lt_s, :rb_x-lt_x, :rb_y-lt_y]
        dice_tempt = dice(pred_map, img_ed_gt.argmax(0))
        dices.append(dice_tempt)
        dices_withname[key] = dice_tempt
        if all_indice:
            jaccard_tempt = jaccard(pred_map,img_ed_gt.argmax(0))
            sensitivity_tempt = sensitivity(pred_map,img_ed_gt.argmax(0))
            hd_95_tempt = hausdorff_distance_95(pred_map,img_ed_gt.argmax(0))
            jaccards.append(jaccard_tempt)
            sensitivitys.append(sensitivity_tempt)
            if not np.isnan(hd_95_tempt):
                hd_95s.append(hd_95_tempt)
    if return_name:
        return np.mean(dices), dices,dices_withname
    if all_indice:
        return np.mean(dices),np.mean(jaccards),np.mean(sensitivitys),np.mean(hd_95s)
    return np.mean(dices), dices


def validation_xshape(encoder,decoder,latent_encoder, dataset, all_indice=False,dsbn=False):
    dices = []
    for key in dataset.keys():
        img_ed = dataset[key]['img']  # (13, 216, 256)
        img_ed_gt = dataset[key]['gt']  # (13, 4, 216, 256)
        patch_size = (128, 128, 128)
        # crop image
        img_ed = np.expand_dims(img_ed, axis=0)
        center_pt = dataset[key]['center'][0]  # random.choice(dataset[key]['center'])
        lt_x = int(max(0, center_pt[0] - patch_size[0] / 2))
        lt_y = int(max(0, center_pt[1] - patch_size[1] / 2))
        lt_s = int(max(0, center_pt[2] - patch_size[2] / 2))
        rb_x = int(min(img_ed.shape[2], lt_x + patch_size[0]))
        rb_y = int(min(img_ed.shape[3], lt_y + patch_size[1]))
        rb_s = int(min(img_ed.shape[1], lt_s + patch_size[2]))
        crop_img = np.zeros((1, 128, 128, 128))
        crop_img[:, :rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y] = img_ed[:, lt_s:rb_s, lt_x:rb_x, lt_y:rb_y]  # in case that crop length < 128
        crop_img_pad = crop_img
        input_data = np.expand_dims(crop_img_pad, axis=0)
        data = torch.from_numpy(input_data).float().cuda()
        out = encoder(data)
        out = latent_encoder(out)
        out = decoder(out)
        crop_output = softmax_helper(out).detach().cpu().numpy()
        # crop_output = model(torch.from_numpy(input_data).float().cuda()).detach().cpu().numpy()
        patch_pred = crop_output.squeeze().argmax(0)
        pred_map = np.zeros(img_ed.shape[1:], dtype=np.int64)
        pred_map[lt_s:rb_s, lt_x:rb_x, lt_y:rb_y] = patch_pred[:rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y]
        dices.append(dice(pred_map, img_ed_gt.argmax(0)))

    return np.mean(dices), dices

def validation_yshape(encoder,decoder, dataset, all_indice=False,dsbn=False):
    dices = []
    jaccards = []
    sensitivitys = []
    hd_95s = []
    for key in dataset.keys():
        img_ed = dataset[key]['img']  # (13, 216, 256)
        img_ed_gt = dataset[key]['gt']  # (13, 4, 216, 256)
        patch_size = (128, 128, 128)
        # crop image
        img_ed = np.expand_dims(img_ed, axis=0)
        center_pt = dataset[key]['center'][0]  # random.choice(dataset[key]['center'])
        lt_x = int(max(0, center_pt[0] - patch_size[0] / 2))
        lt_y = int(max(0, center_pt[1] - patch_size[1] / 2))
        lt_s = int(max(0, center_pt[2] - patch_size[2] / 2))
        rb_x = int(min(img_ed.shape[2], lt_x + patch_size[0]))
        rb_y = int(min(img_ed.shape[3], lt_y + patch_size[1]))
        rb_s = int(min(img_ed.shape[1], lt_s + patch_size[2]))
        crop_img = np.zeros((1, 128, 128, 128))
        crop_img[:, :rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y] = img_ed[:, lt_s:rb_s, lt_x:rb_x, lt_y:rb_y]  # in case that crop length < 128
        crop_img_pad = crop_img
        input_data = np.expand_dims(crop_img_pad, axis=0)
        data = torch.from_numpy(input_data).float().cuda()
        out = encoder(data)
        out = decoder(out)
        crop_output = softmax_helper(out).detach().cpu().numpy()
        # crop_output = model(torch.from_numpy(input_data).float().cuda()).detach().cpu().numpy()
        patch_pred = crop_output.squeeze().argmax(0)
        pred_map = np.zeros(img_ed.shape[1:], dtype=np.int64)
        pred_map[lt_s:rb_s, lt_x:rb_x, lt_y:rb_y] = patch_pred[:rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y]
        dices.append(dice(pred_map, img_ed_gt.argmax(0)))
        if all_indice:
            jaccard_tempt = jaccard(pred_map,img_ed_gt.argmax(0))
            sensitivity_tempt = sensitivity(pred_map,img_ed_gt.argmax(0))
            hd_95_tempt = hausdorff_distance_95(pred_map,img_ed_gt.argmax(0))
            jaccards.append(jaccard_tempt)
            sensitivitys.append(sensitivity_tempt)
            if not np.isnan(hd_95_tempt):
                hd_95s.append(hd_95_tempt)
    if all_indice:
        return np.mean(dices),np.mean(jaccards),np.mean(sensitivitys),np.mean(hd_95s)
    return np.mean(dices), dices

def test_data_yshape(encoder,decoder, dataset,saved_path, all_indice=False,dsbn=False):
    dices = []
    jaccards = []
    sensitivitys = []
    hd_95s = []
    for key in dataset.keys():
        img_ed = dataset[key]['img']  # (13, 216, 256)
        img_ed_gt = dataset[key]['gt']  # (13, 4, 216, 256)
        [affine, header] = dataset[key]['nii_']
        patch_size = (128, 128, 128)
        # crop image
        img_ed = np.expand_dims(img_ed, axis=0)
        center_pt = dataset[key]['center'][0]  # random.choice(dataset[key]['center'])
        lt_x = int(max(0, center_pt[0] - patch_size[0] / 2))
        lt_y = int(max(0, center_pt[1] - patch_size[1] / 2))
        lt_s = int(max(0, center_pt[2] - patch_size[2] / 2))
        rb_x = int(min(img_ed.shape[2], lt_x + patch_size[0]))
        rb_y = int(min(img_ed.shape[3], lt_y + patch_size[1]))
        rb_s = int(min(img_ed.shape[1], lt_s + patch_size[2]))
        crop_img = np.zeros((1, 128, 128, 128))
        crop_img[:, :rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y] = img_ed[:, lt_s:rb_s, lt_x:rb_x, lt_y:rb_y]  # in case that crop length < 128
        crop_img_pad = crop_img
        input_data = np.expand_dims(crop_img_pad, axis=0)
        data = torch.from_numpy(input_data).float().cuda()
        out = encoder(data)
        out = decoder(out)
        crop_output = softmax_helper(out).detach().cpu().numpy()
        # crop_output = model(torch.from_numpy(input_data).float().cuda()).detach().cpu().numpy()
        patch_pred = crop_output.squeeze().argmax(0)
        pred_map = np.zeros(img_ed.shape[1:], dtype=np.int64)
        pred_map[lt_s:rb_s, lt_x:rb_x, lt_y:rb_y] = patch_pred[:rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y]
        dice_tempt = dice(pred_map, img_ed_gt.argmax(0))
        pred_map = np.transpose(pred_map, (1, 2, 0))
        dices.append(dice_tempt)
        save_nii(saved_path + '/{}_{}_pred.nii.gz'.format(key,dice_tempt),pred_map,affine,header)


def test_data(model, dataset, saved_path,dsbn=False):
    dices = []
    for key in dataset.keys():
        img_ed = dataset[key]['img']  # (13, 216, 256)
        img_ed_gt = dataset[key]['gt']  # (13, 4, 216, 256)
        [affine,header] = dataset[key]['nii_']
        patch_size = (128, 128, 128)
        # crop image
        img_ed = np.expand_dims(img_ed, axis=0)
        center_pt = dataset[key]['center'][0]  # random.choice(dataset[key]['center'])
        lt_x = int(max(0, center_pt[0] - patch_size[0] / 2))
        lt_y = int(max(0, center_pt[1] - patch_size[1] / 2))
        lt_s = int(max(0, center_pt[2] - patch_size[2] / 2))
        rb_x = int(min(img_ed.shape[2], lt_x + patch_size[0]))
        rb_y = int(min(img_ed.shape[3], lt_y + patch_size[1]))
        rb_s = int(min(img_ed.shape[1], lt_s + patch_size[2]))
        crop_img = np.zeros((1, 128, 128, 128))
        crop_img[:, :rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y] = img_ed[:, lt_s:rb_s, lt_x:rb_x,
                                                                lt_y:rb_y]  # in case that crop length < 128
        crop_img_pad = crop_img
        input_data = np.expand_dims(crop_img_pad, axis=0)
        if dsbn:
            crop_output = softmax_helper(model(torch.from_numpy(input_data).float().cuda(), [1])).detach().cpu().numpy()
        else:
            crop_output = softmax_helper(model(torch.from_numpy(input_data).float().cuda())).detach().cpu().numpy()
        # crop_output = model(torch.from_numpy(input_data).float().cuda()).detach().cpu().numpy()
        patch_pred = crop_output.squeeze().argmax(0)
        pred_map = np.zeros(img_ed.shape[1:], dtype=np.int64)
        pred_map[lt_s:rb_s, lt_x:rb_x, lt_y:rb_y] = patch_pred[:rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y]
        dice_tempt = dice(pred_map, img_ed_gt.argmax(0))
        pred_map = np.transpose(pred_map,(1,2,0))
        dices.append(dice_tempt)
        save_nii(saved_path + '/{}_{}_pred.nii.gz'.format(key,dice_tempt),pred_map,affine,header)


# import matplotlib.pyplot as plt
def predict(model, img, label=None, num_pool=5):
    padded_img, slicer = pad_img_to_fit_network(img, num_pool)
    if len(padded_img.shape) == 3:
        padded_img = np.expand_dims(padded_img, axis=1)
    padded_output = softmax_helper(model(torch.from_numpy(padded_img).cuda())).detach().cpu().numpy()
    # print(slicer)
    slicer_output = [slicer[0], slice(0, 4, None), slicer[1], slicer[2]]
    output = np.round_(padded_output[tuple(slicer_output)])
    argmax_output = np.argmax(output, axis=1)

    dc = None
    if label is not None:
        if len(np.unique(label)) > 2:
            label = convert_to_one_hot(label)
        dc = dice(output[:, 1:, :, :], label[:, 1:, :, :])
    return argmax_output, output, dc

def make_pseudo_labels(init_label, cur_pred, alpha):
    # def expand_labels(label):
    #     return np.concatenate((1.0 - label, label), axis=1)
    # expanded_init_labels = expand_labels(init_label)
    # expanded_cur_pred = expand_labels(cur_pred)
    y_ = (1 - alpha) * init_label + alpha * cur_pred
    return np.argmax(y_, axis=1).astype(np.uint8)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# result_root = '../results/MultiModality/new_multi_modality_dsbn_kd_fold{}'
# file_list =[]
# for i in range(5):
#     root = result_root.format(i)
#     file_list += os.listdir(root)
# print(file_list)