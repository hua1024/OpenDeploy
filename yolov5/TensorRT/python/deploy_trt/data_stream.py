# coding=utf-8  
# @Time   : 2021/3/23 11:48
# @Auto   : zzf-jeff

import os
import glob
import numpy as np
import cv2
from functools import singledispatch


class CalibStream(object):
    def __init__(self, batch_size, img_size, max_batches, calib_img_dir, mode='det'):
        self.index = 0
        self.length = max_batches
        self.batch_size = batch_size
        self.img_list = glob.glob(os.path.join(calib_img_dir, "*.png"))
        # assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(
        #     calib_img_dir) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))

        self.mode = mode
        self.calibration_data = np.zeros((self.batch_size, *img_size), dtype=np.float32)

        if self.mode == 'det':
            self.normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.to_chw = ToCHWImage()
            self.resize = DetResizeForTest(short_size=736, mode='db')
        elif self.mode == 'rec':
            self.resize = RecResizeImg(image_shape=[3, 32, 800], infer_mode=False, character_type='ch')

    def reset(self):
        self.index = 0

    def data_preprocess(self, img):
        if self.mode == 'det':
            data = {'image': img}
            data = self.resize(data)
            data = self.normalize(data)
            data = self.to_chw(data)
        elif self.mode == 'rec':
            data = {'image': img}
            data = self.resize(data)

        return data['image']

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = self.data_preprocess(img)
                self.calibration_data[i] = img
            self.index += 1
            # example only
            # ascontiguousarray : make data contiguous
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length
