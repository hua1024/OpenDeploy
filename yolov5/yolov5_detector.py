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

from deploy.trt_inference import TRTModel
from utils import preprocess_img, non_max_suppression, scale_coords, xyxy2xywh, plot_one_box


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--weights', type=str, help='trt engine')
    parser.add_argument('--img_path', type=str, help='input image path')
    parser.add_argument('--output_path', type=str, default='output', help='output result path')
    args = parser.parse_args()
    return args


def check_point(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h
    if abs(x2 - x1) > 0.7 * w:
        x2 = w
        x1 = 0
    return x1, y1, x2, y2


class Detector(object):
    def __init__(self, engine_path, conf_thresh=0.1, iou_thresh=0.6):
        super(Detector, self).__init__()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # need change , label index，nc(classes number) , anchors
        self.names = {0: 'table'}
        nc = 1  # classes
        anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ])

        nl = len(anchors)
        a = anchors.copy().astype(np.float32)
        a = a.reshape(nl, -1, 2)
        self.anchor_grid = a.copy().reshape(nl, 1, -1, 1, 1, 2)
        self.no = nc + 5  # outputs per anchor
        self.grid = [torch.zeros(1)] * nl
        self.strides = np.array([8., 16., 32.])

        self.trt = TRTModel(engine_path)

        # warm-up
        self.trt.run(torch.randn(1, 3, 640, 640).numpy())

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def post_process(self, outputs):
        """
        Transforms raw output into boxes, confs, classes
        Applies NMS thresholding on bounding boxes and confs
        Parameters:
            output: raw output tensor
        Returns:
            boxes: x1,y1,x2,y2 tensor (dets, 4)
            confs: class * obj prob tensor (dets, 1)
            classes: class type tensor (dets, 1)
        """
        z = []
        for i in range(len(outputs)):
            outputs[i] = torch.from_numpy(outputs[i])
            if self.grid[i].shape[2:4] != outputs[i].shape[2:4]:
                _, _, height, width, _ = outputs[i].shape
                self.grid[i] = self._make_grid(width, height)
            y = outputs[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.strides[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(1, -1, self.no))

        pred = torch.cat(z, 1)
        pred = non_max_suppression(pred, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh)
        return pred

    def run(self, img):
        # 嵌入部分表格检测的业务
        res_list = []
        img_list = []
        resize_img = preprocess_img(img)
        output = self.trt.run(resize_img)
        pred = self.post_process(output)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(resize_img.shape[2:], det[:, :4], img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if float('%.2f' % conf) > self.conf_thresh:
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        # add by lcw
                        # x1, y1, x2, y2 = check_point(img, x1, y1, x2, y2)
                        # label is self.names[int(cls)], score is conf
                        res_list.append((x1, y1, x2, y2))
                        img_list.append(img[y1:y2, x1:x2])

        return res_list, img_list


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
    detector = Detector(engine_path=weights, conf_thresh=0.7, iou_thresh=0.6)
    # Run inference
    # while 1:
    for file in tqdm(img_list):
        basename = os.path.basename(file)
        img = cv2.imread(file)
        res_list, imgs_list = detector.run(img)
        for res in res_list:
            with open(os.path.join(output_path, os.path.splitext(basename)[0] + '_trt.txt'), 'a') as f:
                f.write(('%g, ' * 4 + '\n') % (res))  # label format
            label = '%s %.2f' % ('table', 0.0)
            plot_one_box(res, img, label=label)
        cv2.imwrite(os.path.join(output_path, basename), img)


if __name__ == '__main__':
    main()
