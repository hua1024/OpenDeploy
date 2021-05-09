# coding=utf-8  
# @Time   : 2021/3/6 9:12
# @Auto   : zzf-jeff
import sys, os

sys.path.append('./')
import argparse
from typing import Tuple, List
import torch
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import time
from deploy.trt_inference import TRTModel


def test1():
    from torchvision import transforms
    from PIL import Image
    import time

    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(size=[320, 320]),
        transforms.ToTensor(),
        normalize_imgnet
    ])

    img = Image.open('imgs/1.png')
    img = img.convert('RGB')
    img = trans(img)
    img = img.unsqueeze(0)
    img_numpy = np.array(img, dtype=np.float32, order='C')

    trt = TRTModel('weights/dc-5x-sim.engine')
    for _ in range(10):
        start_time = time.time()
        output = trt.run(img_numpy)
        output_data = torch.Tensor(output)
        end_time = time.time()

    print(end_time-start_time)
    print(output_data.shape)


test1()
