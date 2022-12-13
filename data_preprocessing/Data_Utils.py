import numpy as np
import nibabel as nib
from medpy.metric import dc, hd
from sklearn.model_selection import KFold
from collections import OrderedDict
import random

def convert_to_one_hot(seg):    # (slices, width, height)
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)    # (channels, slices, width, height)
    for c in range(len(vals)):
        # print('number:',c)
        res[c][seg == c] = 1
    # res = np.moveaxis(res, 0, 1)    # convert to (slices, channels, width, height)
    return res

def convert_to_onehot(seg):    # (slices, width, height)
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)    # (channels, slices, width, height)
    for c in range(len(vals)):
        # print('number:',c)
        res[c][seg == c] = 1
    # res = np.moveaxis(res, 0, 1)    # convert to (slices, channels, width, height)
    return res

def normalize_img(img, eps=1e-8,norm_type='z-score'):
    if norm_type == 'z-score':
        m = np.mean(img)
        std = np.std(img)
        result = (img - m + eps) / (std + eps)
    elif norm_type == 'max-abs':
        # print('.........')
        max_ = np.max(np.abs(img))
        result = img / max_
    elif norm_type == 'max-min':
        clip_max = np.percentile(img,90)
        clip_min = np.percentile(img,10)
        img = np.clip(img,clip_min,clip_max)
        max_ = np.max(img)
        min_ = np.min(img)
        print('max-min:',max_,min_,np.mean(img))
        # result = ((img - min_ + eps) // (max_ - min_ + eps))
        result = img
    else:
        result = img
    return result


def load_nii(img_path, reorient=False):
    nimg = nib.load(img_path)
    if reorient:
        nimg = nib.as_closest_canonical(nimg)
    return nimg.get_data(), nimg.affine, nimg.header


def get_orientation(affine):
    return nib.aff2axcodes(affine)

def center_crop_3D_image_batched(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape))
    else:
        assert len(crop_size) == len(
            img.shape) , "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    lt_x = max(0, crop_size[0] // 2 + random.randint(-30, 30))
    lt_y = max(0, crop_size[1] // 2 + random.randint(-30, 30))
    lt_s = max(0, crop_size[2] // 2 + random.randint(-30, 30))
    rb_x = min(img.shape[1], lt_x + crop_size[0])
    rb_y = min(img.shape[2], lt_y + crop_size[1])
    rb_s = min(img.shape[0], lt_s + crop_size[2])
    return img[lt_s:rb_s,lt_x:rb_x,lt_y:rb_y]

def center_crop_3D_image_gt(img,gt, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape))
    else:
        assert len(crop_size) == len(
            img.shape) , "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    lt_x = max(0, crop_size[0] // 2 + random.randint(-30, 30))
    lt_y = max(0, crop_size[1] // 2 + random.randint(-30, 30))
    lt_s = max(0, crop_size[2] // 2 + random.randint(-30, 30))
    rb_x = min(img.shape[1], lt_x + crop_size[0])
    rb_y = min(img.shape[2], lt_y + crop_size[1])
    rb_s = min(img.shape[0], lt_s + crop_size[2])
    return img[lt_s:rb_s,lt_x:rb_x,lt_y:rb_y],gt[lt_s:rb_s,lt_x:rb_x,lt_y:rb_y]


def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def windwo_transform(ct_arry,wind_width,wind_center):
    min_window = float(wind_center) - 0.5 * float(wind_width)
    max_window = float(wind_center) + 0.5 * float(wind_width)
    ct_arry = np.clip(ct_arry,min_window,max_window)
    return ct_arry

def adjust_border(center,shape,crop_size):
    lt = max(0,center - crop_size)
    rt = min(shape,center + crop_size)
    if lt == 0:
        rt = lt + crop_size * 2
    elif rt == shape:
        lt = rt - crop_size * 2
    return lt,rt

def roi_crop_data(img,gt,crop_size=[256,256,256]):
    args = np.argwhere(gt == 1)
    min_x,max_x,min_y,max_y,min_z,max_z = min(args[:, 0]), max(args[:, 0]), min(args[:, 1]), max(args[:, 1]), min(args[:, 2]), max(args[:, 2])
    center_x,center_y,center_z = int((max_x+min_x)/2),int((max_y+min_y)/2),int((max_z+min_z)/2)
    print(min(args[:, 0]), max(args[:, 0]), min(args[:, 1]), max(args[:, 1]), min(args[:, 2]), max(args[:, 2]))
    print(center_x, center_y, center_z)
    lt_x,rb_x = adjust_border(center_x,img.shape[0],crop_size[0] // 2)
    lt_y, rb_y = adjust_border(center_y, img.shape[1], crop_size[1] // 2)
    lt_s, rb_s = adjust_border(center_z, img.shape[2], crop_size[2] // 2)
    return img[lt_x:rb_x,lt_y:rb_y,lt_s:rb_s],gt[lt_x:rb_x,lt_y:rb_y,lt_s:rb_s]

def normalize_img_after_windowtransform(img,window_center,window_width,eps=1e-8,norm_type = 'z-score'):
    m = np.mean(img)
    std = np.std(img)
    min_window = float(window_center) - 0.5 * float(window_width)
    max_window = float(window_center) + 0.5 * float(window_width)
    img_mask = img[np.where((img > min_window)&(img < max_window))]
    # print("mask shape:",img_mask.shape)
    if img_mask.shape[0] != 0:
        m = np.mean(img_mask)
        std = np.std(img_mask)
    if norm_type == 'z-score':
        m = np.mean(img_mask)
        std = np.std(img_mask)
        result = (img - m + eps) / (std + eps)
    elif norm_type == 'max-abs':
        # print('.........')
        max_ = np.max(np.abs(img_mask))
        result = img / max_
    elif norm_type == 'max-min':
        max_ = np.max(img_mask)
        min_ = np.min(img_mask)
        result = ((img - min_ + eps) // (max_ - min_ + eps)) * 2 - 1
    else:
        result = img
    # print("m and std:",m,std)
    return result

def metrics(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, volpred, volpred-volgt]

    return res


def split_data(patient_list, K=5, shuffle=False):
    """
    :param patient_list:
    :param K: K-fold cross-validation
    :return: 5 train-val pairs
    """
    splits = []
    # sort patient_list to ensure the splited data unchangeable every time.
    patient_list.sort()
    # k-fold, I think it doesn't matter whether the shuffle is true or not
    kfold = KFold(n_splits=K, shuffle=shuffle, random_state=12345)
    for i, (train_idx, test_idx) in enumerate(kfold.split(patient_list)):
        train_keys = np.array(patient_list)[train_idx]
        test_keys = np.array(patient_list)[test_idx]
        splits.append(OrderedDict())
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = test_keys
    return splits

