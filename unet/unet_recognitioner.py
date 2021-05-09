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
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from utils import preprocess_img
from deploy.trt_inference import TRTModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--weights', type=str, help='trt engine')
    parser.add_argument('--img_path', type=str, help='input image path')
    parser.add_argument('--output_path', type=str, default='output', help='output result path')
    args = parser.parse_args()
    return args


class Recognitioner(object):
    def __init__(self, engine_path, scale_factor, conf_thresh=0.5):
        super(Recognitioner, self).__init__()
        self.conf_thresh = conf_thresh
        self.scale_factor = scale_factor
        # need change , label index，nc(classes number)
        self.nc = 3
        self.trt = TRTModel(engine_path)
        # warm-up
        self.trt.run(torch.randn(1, 3, 500, 500).numpy())

    def post_process(self, outputs, img_shape):
        output = torch.from_numpy(outputs[0])
        if self.nc > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_shape[1]),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        return full_mask

    def mask_to_image(self, mask):
        if len(mask.shape) == 3:
            mask = torch.from_numpy(mask)
            probs = mask.max(0)[0]
            type_prob = mask.max(0)[1]
            row_mask = type_prob == 1
            col_mask = type_prob == 2
            probs = torch.where(row_mask > 0, torch.ones_like(probs) * 128, torch.zeros_like(probs))
            probs = torch.where(col_mask > 0, torch.ones_like(probs) * 255, probs)
            probs = probs.numpy()
            return Image.fromarray((probs).astype(np.uint8))
        else:
            return Image.fromarray((mask * 255).astype(np.uint8))

    def run(self, img):
        resize_img = preprocess_img(img, self.scale_factor)
        output = self.trt.run(resize_img)
        full_mask = self.post_process(output, img.size)
        # full_mask = full_mask > self.conf_thresh
        mask_image = self.mask_to_image(full_mask)
        return mask_image


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
    args = parse_args()
    weights = args.weights
    img_path = args.img_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_list = load_img_by_path(img_path)
    detector = Recognitioner(engine_path=weights, scale_factor=0.5, conf_thresh=0.5)
    # Run inference
    # while 1:
    for file in tqdm(img_list):
        basename = os.path.basename(file)
        img = Image.open(file)
        mask_image = detector.run(img)
        mask_image.save(os.path.join(output_path, basename))


if __name__ == '__main__':
    main()
