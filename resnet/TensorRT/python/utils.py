# coding=utf-8  
# @Time   : 2021/3/26 10:34
# @Auto   : zzf-jeff

import cv2
import numpy as np
import torchvision
import time
import torch
import random


def plot_label(x, img, label, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a point likes (x1,y1)
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1]))

    tf = max(tl - 1, 1)  # font thickness
    cv2.putText(
        img,
        label,
        (c1, c2),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )


def his_imnormalize(img, mean, std, to_rgb=True):
    # 同步海思用的归一化
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = img.copy().astype(np.float32)
    assert img.dtype != np.uint8
    # mean = np.float64(mean.reshape(1, -1))
    # stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # inplace
        img = img / 255.
    # cv2.subtract(img, mean, img)  # inplace
    # cv2.multiply(img, stdinv, img)  # inplace
    # img /= 255.0
    return img


def preprocess_img(img, mean, std, to_rgb=True):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = his_imnormalize(img, mean, std, to_rgb)
    # squeeze channel (224,224,3) --> (1,224,224,3)
    img = np.expand_dims(img, axis=0)
    # transpose chanel (1,224,224,3) --> (1,3,224,224)
    img = img.transpose(0, 3, 1, 2)
    img = np.array(img, dtype=np.float32, order='C')
    return img
