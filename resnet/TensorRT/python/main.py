# coding=utf-8  
# @Time   : 2021/3/26 15:13
# @Auto   : zzf-jeff

import argparse
import torch
import numpy as np
import time
import cv2
import os
from tqdm import tqdm

from config import opt
from deploy_trt.trt_inference import TRTModel
from deploy_trt.onnx_inference import ONNXModel
from utils import preprocess_img, plot_label


class Classifier(object):
    """resnet inference code (resnet推理代码)
    support model type is torch/onnx/tensorrt

    """

    def __init__(self, model_path, label_path, conf_thresh=0.1, mean=None, std=None):

        super(Classifier, self).__init__()
        self.conf_thresh = conf_thresh
        self.mean = mean
        self.std = std
        # read label dict
        with open(label_path, 'r', encoding='utf-8') as f:
            self.names = f.read().rstrip('\n').split('\n')
        # init model

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._init_model(model_path)

    def _init_model(self, model_path):
        self.mode = 'torch'
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            self.mode = 'torch'
            # self.model = torch.load(model_path, map_location='cpu').eval()
        elif model_path.endswith('.onnx'):
            self.mode = 'onnx'
            self.model = ONNXModel(model_path)
        elif model_path.endswith('.engine'):
            self.mode = 'trt'
            self.model = TRTModel(model_path)

        warmup_data = torch.randn(1, 3, 224, 224).to(self.device)
        self.model(warmup_data)  # warm-up
        del warmup_data

    def post_process(self, outputs):
        dst_dict = {}
        outputs = outputs[0].cpu().numpy()
        pred_score = np.max(outputs, axis=1)[0]
        pred_label_idx = np.argmax(outputs, axis=1)[0]
        pred_label = self.names[pred_label_idx]
        dst_dict.update({"pred_score": pred_score, "pred_label": pred_label})
        return dst_dict

    def run(self, img):
        '''yolov5 trt inference func

        :param img: np img
        :return:
            dst_list : [(x1,y1,x2,y2,label,conf),...]
        '''
        dst_list = []
        # pre process
        resize_img = preprocess_img(img, self.mean, self.std)
        resize_img = torch.from_numpy(resize_img).to(self.device)
        # inference
        output = self.model(resize_img)
        # post process
        pred = self.post_process(output)
        return pred


def load_img_by_path(path):
    res_list = []
    if not os.path.isfile(path):
        for fpathe, dirs, fs in os.walk(path):
            for f in fs:
                temp = os.path.join(fpathe, f)
                if 'visual' not in temp:
                    if temp.endswith('png') or temp.endswith('jpg'):
                        res_list.append(temp)
    else:
        res_list.append(path)
    return res_list


def main():
    # get info
    model_path = opt.model_path
    img_path = opt.img_path
    output_path = opt.output_path
    label_path = opt.label_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_list = load_img_by_path(img_path)

    classifier = Classifier(
        model_path=model_path,
        label_path=label_path,
        conf_thresh=opt.conf_thresh, )
    # Run inference
    # test
    # while True:
    for file in tqdm(img_list):
        basename = os.path.basename(file)
        img = cv2.imread(file)
        pred_dict = classifier.run(img)
        label = '%s %.2f' % (pred_dict["pred_label"], pred_dict["pred_score"])
        with open(os.path.join(output_path, os.path.splitext(basename)[0] + '.txt'), 'a+') as f:
            f.write(('%s' + '\n') % (label))  # label format

        plot_label((30, 30), img, label)
        cv2.imwrite(os.path.join(output_path, basename), img)


if __name__ == '__main__':
    main()
