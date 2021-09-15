# coding=utf-8  
# @Time   : 2021/3/6 10:33
# @Auto   : zzf-jeff

# other env without trt
# without torch , using cuda.mem_alloc,


try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt
except:
    print('trt need env error!!!')

import re
import numpy as np
import time


class TRTModel():

    def __init__(self, engine_path):
        """Tensorrt engine model dynamic inference

        Args:
            engine_path (trt.tensorrt.ICudaEngine)
        """
        super(TRTModel, self).__init__()
        # cfx多线程需要加的限制
        self.cfx = pycuda.autoinit.context
        self.engine_path = engine_path
        self.logger = trt.Logger(getattr(trt.Logger, 'ERROR'))
        ## load engine for engine_path
        self.engine = self.load_engine()
        self.stream = cuda.Stream()
        # default profile index is 0
        self.profile_index = 0
        ## create context for cuda engine
        self.context = self.engine.create_execution_context()
        self.batch_size_ranges = []
        ## get input/deploy_trtoutput cuda swap address use idx
        self.input_binding_idxs, self.output_binding_idxs = self._get_binding_idxs()
        ## get network input/output name
        self.input_names, self.output_names = self.get_input_output_name()

    def __del__(self):
        del self.engine
        del self.stream
        del self.context

    def load_engine(self):
        '''Load cuda engine

        :return: engine
        '''
        print("Loaded engine: {}".format(self.engine_path))
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _get_binding_idxs(self, profile_index=0):
        """

        :param engine:
        :param profile_index:
        :return:
        """
        # Calculate start/end binding indices for current context's profile
        num_bindings_per_profile = self.engine.num_bindings // self.engine.num_optimization_profiles
        start_binding = profile_index * num_bindings_per_profile
        end_binding = start_binding + num_bindings_per_profile
        # Separate input and output binding indices for convenience
        input_binding_idxs = []
        output_binding_idxs = []
        for binding_index in range(start_binding, end_binding):
            if self.engine.binding_is_input(binding_index):
                input_binding_idxs.append(binding_index)
            else:
                output_binding_idxs.append(binding_index)
        return input_binding_idxs, output_binding_idxs

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

    # 指定输入的shape，同时根据输入的shape指定输出的shape，并未输出赋予cuda空间
    def _get_bindings(self):
        # Explicitly set the dynamic input shapes, so the dynamic output
        # shapes can be computed internally
        host_outputs = []
        device_outputs = []
        for binding_index in self.output_binding_idxs:
            output_shape = self.context.get_binding_shape(binding_index)
            # Allocate buffers to hold output results after copying back to host
            buffer = np.empty(output_shape, dtype=np.float32)
            host_outputs.append(buffer)
            # Allocate output buffers on device
            device_outputs.append(cuda.mem_alloc(buffer.nbytes))

        return host_outputs, device_outputs

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

    def _activate_profile(self, batch_size):
        for idx, bs_range in enumerate(self.batch_size_ranges):
            if bs_range[0] <= batch_size <= bs_range[1]:
                if self.profile_index != idx:
                    self.profile_index = idx
                    # 动态推理需要设定profile idx，为了进行不同的内存分配
                    self.context.active_optimization_profile = idx

    def _flatten(self, inp):
        if not isinstance(inp, (tuple, list)):
            return [inp]
        out = []
        for sub_inp in inp:
            out.extend(self._flatten(sub_inp))

        return out

    def __call__(self, inputs):
        ''' pycuda trt推理

        Args:
            image:
        Returns:
        '''
        self.cfx.push()
        host_inputs = self._flatten(inputs)
        batch_size = host_inputs[0].shape[0]
        assert batch_size <= self.engine.max_batch_size, (
            'input batch_size {} is larger than engine max_batch_size {}, '
            'please increase max_batch_size and rebuild engine.'
        ).format(batch_size, self.engine.max_batch_size)
        # support dynamic batch size when engine has explicit batch dimension.
        if not self.engine.has_implicit_batch_dimension:
            self._activate_profile(batch_size)
            self._set_binding_shape(host_inputs)

        # Allocate device memory for inputs. This can be easily re-used if the
        device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]

        # Copy host inputs to device, this needs to be done for each new input， 由host拷贝到device
        for h_input, d_input in zip(host_inputs, device_inputs):
            cuda.memcpy_htod_async(d_input, h_input, self.stream)

        host_outputs, device_outputs = self._get_bindings()
        # Bindings are a list of device pointers for inputs and outputs
        bindings = device_inputs + device_outputs  # list的合并

        # Inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Copy outputs back to host to view results 将输出由gpu拷贝到cpu。
        outputs = []

        for h_output, d_output in zip(host_outputs, device_outputs):
            cuda.memcpy_dtoh_async(h_output, d_output, self.stream)
            outputs.append(h_output.reshape(h_output.shape))

        self.stream.synchronize()

        self.cfx.pop()

        return outputs
