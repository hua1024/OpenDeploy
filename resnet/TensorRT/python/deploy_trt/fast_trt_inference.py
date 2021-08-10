# coding=utf-8  
# @Time   : 2021/3/6 10:33
# @Auto   : zzf-jeff

try:
    import tensorrt as trt
    import torch
except:
    print('trt need env error!!!')

import re
import numpy as np
import time
from torch import nn


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        raise TypeError('{} is not supported by torch'.format(device))


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('{} is not supported by torch'.format(dtype))


class TRTModel(nn.Module):

    def __init__(self, engine_path):
        """Tensorrt engine model dynamic inference

        Args:
            engine_path (trt.tensorrt.ICudaEngine)
        """
        super(TRTModel, self).__init__()
        self.engine_path = engine_path
        self.logger = trt.Logger(getattr(trt.Logger, 'ERROR'))

        ## load engine for engine_path
        self.engine = self.load_engine()
        ## gen context
        self.context = self.engine.create_execution_context()

        ##get network input and output names
        self.input_names, self.output_names = self.get_input_output_name()

        # get batch size range of each profile
        self.batch_size_ranges = []
        for idx in range(self.engine.num_optimization_profiles):
            name = self._rename(idx, self.input_names[0])
            min_shape, opt_shape, max_shape = self.engine.get_profile_shape(
                idx, name)
            self.batch_size_ranges.append((min_shape[0], max_shape[0]))

        # default profile index is 0
        self.profile_index = 0

    def load_engine(self):
        '''Load cuda engine

        :return: engine
        '''
        print("Loaded engine: {}".format(self.engine_path))
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_output_name(self):
        """
        :return:
        """
        # get engine input tensor names and output tensor names
        input_names, output_names = [], []
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            if not re.match(r'.* \[profile \d+\]', name):
                if self.engine.binding_is_input(idx):
                    input_names.append(name)
                else:
                    output_names.append(name)
        return input_names, output_names

    def _activate_profile(self, batch_size):
        for idx, bs_range in enumerate(self.batch_size_ranges):
            if bs_range[0] <= batch_size <= bs_range[1]:
                if self.profile_index != idx:
                    self.profile_index = idx
                    # 动态推理需要设定profile idx，为了进行不同的内存分配
                    self.context.active_optimization_profile = idx

    @staticmethod
    def _rename(idx, name):
        if idx > 0:
            name += ' [profile {}]'.format(idx)
        return name

    def _set_binding_shape(self, inputs):
        for name, inp in zip(self.input_names, inputs):
            name = self._rename(self.profile_index, name)
            idx = self.engine.get_binding_index(name)
            binding_shape = tuple(self.context.get_binding_shape(idx))
            shape = tuple(inp.shape)
            if shape != binding_shape:
                self.context.set_binding_shape(idx, shape)

    @property
    def input_length(self):
        return len(self.input_names)

    @property
    def output_length(self):
        return len(self.output_names)

    @property
    def total_length(self):
        return self.input_length + self.output_length

    def _get_bindings(self, inputs):
        bindings = [None] * self.total_length
        outputs = [None] * self.output_length

        for i, name in enumerate(self.input_names):
            name = self._rename(self.profile_index, name)
            idx = self.engine.get_binding_index(name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))

            bindings[idx % self.total_length] = (
                inputs[i].to(dtype).contiguous().data_ptr())

        for i, name in enumerate(self.output_names):
            name = self._rename(self.profile_index, name)
            idx = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(idx))
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(
                size=shape, dtype=dtype, device=device).contiguous()
            outputs[i] = output
            bindings[idx % self.total_length] = output.data_ptr()

        return outputs, bindings

    def _flatten(self, inp):
        if not isinstance(inp, (tuple, list)):
            return [inp]
        out = []
        for sub_inp in inp:
            out.extend(self._flatten(sub_inp))

        return out

    def forward(self, inputs):
        inputs = self._flatten(inputs)
        batch_size = inputs[0].shape[0]
        assert batch_size <= self.engine.max_batch_size, (
            'input batch_size {} is larger than engine max_batch_size {}, '
            'please increase max_batch_size and rebuild engine.'
        ).format(batch_size, self.engine.max_batch_size)

        # support dynamic batch size when engine has explicit batch dimension.
        if not self.engine.has_implicit_batch_dimension:
            self._activate_profile(batch_size)
            self._set_binding_shape(inputs)

        outputs, bindings = self._get_bindings(inputs)
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)

        return outputs
