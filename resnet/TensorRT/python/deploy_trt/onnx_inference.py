# coding=utf-8  
# @Time   : 2020/12/3 17:31
# @Auto   : zzf-jeff

'''
onnx model inference
using to check onnx output equal torch output
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import onnxruntime
import numpy as np
from torch import nn
import torch


class ONNXModel(nn.Module):

    def __init__(self, onnx_path):
        """Onnx Model inference function

        Args:
            onnx_path: onnx model path
        """
        super().__init__()
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    def __del__(self):
        del self.onnx_session
        del self.input_name
        del self.output_name

    def to_numpy(self, data):
        if not isinstance(data, np.ndarray):
            if data.requires_grad:
                data = data.detach().cpu().numpy()
            else:
                data = data.cpu().numpy()
        return data

    def get_output_name(self, onnx_session):
        """ 根据onnx session获取输出name
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """根据onnx session获取输入name
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """ 组装需要的输入格式
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image):
        input_feed = self.get_input_feed(self.input_name, self.to_numpy(image))
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        # keep output
        return [torch.from_numpy(out) for out in outputs]
