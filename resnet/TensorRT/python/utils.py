# coding=utf-8  
# @Time   : 2021/3/26 10:34
# @Auto   : zzf-jeff

import cv2
import time
import torch
import os


def plot_classify_label(x, img, label, line_thickness=None):
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


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def split_batch(one_data_list, batch_size=4):
    # list split by idx,using to split batch
    return [one_data_list[i:i + batch_size] for i in range(len(one_data_list)) if i % batch_size == 0]


def load_img_data(path):
    img_formats = ['jpg', 'jpeg', 'png', 'bmp', 'JPEG', 'PNG', 'JPG']  # match img 后缀
    res_list = []
    if os.path.isdir(path):
        for dir_path, dir_names, filenames in os.walk(path):
            for filename in filenames:
                temp_path = os.path.join(dir_path, filename)
                res_list.append(temp_path)
    elif os.path.isfile(path):
        res_list.append(path)

    res_list = [x for x in res_list if x.split('.')[-1].lower() in img_formats]
    return res_list


def load_img_data_batch(path, bs=1):
    img_formats = ['jpg', 'jpeg', 'png', 'bmp', 'JPEG', 'PNG', 'JPG']  # match img 后缀
    res_list = []
    if os.path.isdir(path):
        for dir_path, dir_names, filenames in os.walk(path):
            for filename in filenames:
                temp_path = os.path.join(dir_path, filename)
                res_list.append(temp_path)
    elif os.path.isfile(path):
        res_list.append(path)

    res_list = [x for x in res_list if x.split('.')[-1].lower() in img_formats]
    res_list = split_batch(res_list, batch_size=bs)

    return res_list
