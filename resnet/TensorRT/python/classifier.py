# coding=utf-8  
# @Time   : 2020/12/29 18:32
# @Auto   : zzf-jeff

import torch
import numpy as np
import cv2

from deploy_trt.fast_trt_inference import TRTModel
# from deploy_trt.slow_trt_inference import TRTModel
from deploy_trt.onnx_inference import ONNXModel


class Classifier(object):
    """resnet inference code (resnet推理代码)
    support model type is onnx/tensorrt

    """

    def __init__(self, model_path, label_path, img_size=(224,224), conf_thresh=0.1, mean=None, std=None):

        super(Classifier, self).__init__()
        self.conf_thresh = conf_thresh
        self.mean = mean
        self.std = std
        self.img_size = img_size
        # read label dict
        with open(label_path, 'r', encoding='utf-8') as f:
            self.names = f.read().rstrip('\n').split('\n')
        # init model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._init_model(model_path)

    def _init_model(self, model_path):
        self.mode = 'trt'
        if model_path.endswith('.onnx'):
            self.mode = 'onnx'
            self.model = ONNXModel(model_path)
        elif model_path.endswith('.engine'):
            self.mode = 'trt'
            self.model = TRTModel(model_path)

        for _ in range(10):
            # warmup_data = torch.randn(1, 3, 224, 224).numpy()
            warmup_data = torch.randn(1, 3, 224, 224).to(self.device)
            self.model(warmup_data)  # warm-up
        del warmup_data

    def his_imnormalize(self, img, mean, std, to_rgb=True):
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

    def preprocess_img(self, img, mean, std, to_rgb=True):
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        img = self.his_imnormalize(img, mean, std, to_rgb)
        # squeeze channel (224,224,3) --> (1,224,224,3)
        img = np.expand_dims(img, axis=0)
        # transpose chanel (1,224,224,3) --> (1,3,224,224)
        img = img.transpose(0, 3, 1, 2)
        img = np.array(img, dtype=np.float32, order='C')
        return img

    def preprocess_img_batch(self, inputs, mean, std, to_rgb=True):
        return [self.preprocess_img(input, mean, std, to_rgb) for input in inputs]

    def post_process(self, outputs):
        dst_list = []
        dst_dict = {}
        # outputs = outputs[0]
        outputs = outputs[0].detach().cpu().numpy()
        for output in outputs:
            pred_score = np.max(output, axis=0)
            pred_label_idx = np.argmax(output, axis=0)
            pred_label = self.names[int(pred_label_idx)]
            dst_dict.update({"pred_score": pred_score, "pred_label": pred_label})
            dst_list.append(dst_dict)

        return dst_list

    def run(self, inputs):
        ''' trt inference func

        :param img:[img1,img2]
        :return:
            dst_list : [{pred1},{pred2}]
        '''
        # pre process
        inputs = self.preprocess_img_batch(inputs, self.mean, self.std)
        inputs = np.vstack(inputs)  # cat batch data
        inputs = torch.from_numpy(inputs).to(self.device)
        with torch.no_grad():
            # inference
            output = self.model(inputs)
            # post process
            preds = self.post_process(output)

        return preds
