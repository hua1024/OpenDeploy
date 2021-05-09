# coding=utf-8  
# @Time   : 2021/3/26 10:34
# @Auto   : zzf-jeff

import cv2
import numpy as np
import torchvision
import time
import torch
import random


def preprocess_img(img, scale_factor):
    w, h = img.size
    newW, newH = int(scale_factor * w), int(scale_factor * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = img.resize((newW, newH))
    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    img_trans = img_nd.transpose((2, 0, 1))
    img_trans = img_trans.astype(np.float32)
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    img_trans = np.expand_dims(img_trans, axis=0)
    # Convert the image to row-major order, also known as "C order":
    img_trans = np.ascontiguousarray(img_trans)
    return img_trans
