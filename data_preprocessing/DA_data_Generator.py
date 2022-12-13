from skimage import transform
import glob
import os
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.augmentations.utils import pad_nd_image
import numpy as np
import random


class MutiScale_DataGenerator3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size=(128, 128, 128), batch_size=1,scales = 3):
        super(MutiScale_DataGenerator3D, self).__init__(data, batch_size, None)
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.scales = scales

    def generate_train_batch(self):
        datas = []
        segs = []
        px,py,pz = self.patch_size
        #得到不同尺度的数据
        for i in range(self.scales):
            scale = 2**i
            datas.append(np.zeros((self.batch_size, 1, px // scale, py // scale, pz // scale), dtype=np.float32))
            segs.append(np.zeros((self.batch_size, 1, px // scale, py // scale, pz // scale), dtype=np.float32))
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        for i, key in enumerate(selected_keys):
            img_instance = self._data[key]['img']
            seg_instance = self._data[key]['gt']
            # expand dimensions
            img_instance = np.expand_dims(np.expand_dims(img_instance, axis=0), axis=0)  # (1, 1, slices, width, height)
            seg_instance = np.expand_dims(seg_instance, axis=0)  # (1, num_classes, slices, width, height)
            # stack data and seg
            stacked_img_seg = np.concatenate((img_instance, seg_instance),
                                             axis=1)  # (1, num_classes+1, slices, width, height)
            # pad img in case the shape is smaller than patch_size
            padded_instance = pad_nd_image(stacked_img_seg, self.patch_size)  # (1, num_classes+1, *, *)
            # randomly crop around the center point
            center_pt = random.choice(self._data[key]['center'])  # some may contain more than one centers
            lt_x = max(0, center_pt[0] - 64 + random.randint(-30, 30))
            lt_y = max(0, center_pt[1] - 64 + random.randint(-30, 30))
            lt_s = max(0, center_pt[2] - 64 + random.randint(-30, 30))
            rb_x = min(padded_instance.shape[3], lt_x + 128)
            rb_y = min(padded_instance.shape[4], lt_y + 128)
            rb_s = min(padded_instance.shape[2], lt_s + 128)

            cropped_instance = padded_instance[:, :, lt_s:rb_s, lt_x:rb_x, lt_y:rb_y]
            cropped_instance = pad_nd_image(cropped_instance, self.patch_size)
            # cropped_instance = random_crop_3D_image_batched(padded_instance, self.patch_size)  # (1, num_classes+1, *self.patch_size)

            # data[i, 0] = cropped_instance[0, 0]
            img = cropped_instance[0, 0]
            datas[0] = img
            seg = np.argmax(cropped_instance[0, 1:], axis=0)
            segs[0] = seg
            for i in  range(self.scales - 1):
                scale = 2 ** (i+1)
                datas[i+1] = transform.resize(img, (px // scale, py // scale, pz // scale), order=3, mode='edge', preserve_range=True)
                segs[i+1] = transform.resize(img, (px // scale, py // scale, pz // scale), order=0, mode='edge', preserve_range=True)


        return {'data': datas, 'seg': segs, 'keys': selected_keys}
