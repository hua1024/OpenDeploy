# coding=utf-8  
# @Time   : 2021/3/6 10:33
# @Auto   : zzf-jeff

# other env without trt
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt
except:
    print('trt need env error!!!')
    pass

import re
import numpy as np


class TRTModel():

    def __init__(self, engine_path):
        """Tensorrt engine model dynamic inference

        Args:
            engine_path (trt.tensorrt.ICudaEngine)
        """
        super(TRTModel, self).__init__()
        # cfx多线程需要加的限制
        # self.cfx = cuda.Device(0).make_context()

        self.engine_path = engine_path
        self.logger = trt.Logger(getattr(trt.Logger, 'ERROR'))
        ## load engine for engine_path
        self.engine = self.load_engine()
        self.stream = cuda.Stream()
        ## create context for cuda engine
        self.context = self.engine.create_execution_context()
        ## set activate optimization profile
        self.context.active_optimization_profile = 0
        print("Active Optimization Profile: {}".format(self.context.active_optimization_profile))

        ## get input/output cuda swap address use idx
        self.input_binding_idxs, self.output_binding_idxs = self.get_binding_idxs(self.engine,
                                                                                  self.context.active_optimization_profile)
        ## get network input/output name
        self.input_names, self.output_names = self.get_input_output_name()

        self.is_dynamic = False
        ## judge input is dynamic
        self.print_inputs_info(self.engine, self.context, self.input_binding_idxs)

        ## if not dynamic , set fix host_output、device_outputs、device_inputs
        if not self.is_dynamic:
            self.fix_host_outputs, self.fix_device_outputs, self.fix_device_inputs = self.get_fix_binding_shapes(
                self.context, self.input_binding_idxs, self.output_binding_idxs)

    def load_engine(self):
        '''Load cuda engine

        :return: engine
        '''
        print("Loaded engine: {}".format(self.engine_path))
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_binding_idxs(self, engine, profile_index):
        """

        :param engine:
        :param profile_index:
        :return:
        """
        # Calculate start/end binding indices for current context's profile
        num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
        start_binding = profile_index * num_bindings_per_profile
        end_binding = start_binding + num_bindings_per_profile
        print("Engine/Binding Metadata")
        print("\tNumber of optimization profiles: {}".format(engine.num_optimization_profiles))
        print("\tNumber of bindings per profile: {}".format(num_bindings_per_profile))
        print("\tFirst binding for profile {}: {}".format(profile_index, start_binding))
        print("\tLast binding for profile {}: {}".format(profile_index, end_binding - 1))
        # Separate input and output binding indices for convenience
        input_binding_idxs = []
        output_binding_idxs = []
        for binding_index in range(start_binding, end_binding):
            if engine.binding_is_input(binding_index):
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
    def setup_binding_shapes(
            self,
            context,
            host_inputs,
            input_binding_idxs,
            output_binding_idxs,
    ):
        # Explicitly set the dynamic input shapes, so the dynamic output
        # shapes can be computed internally

        for host_input, binding_index in zip(host_inputs, input_binding_idxs):
            context.set_binding_shape(binding_index, host_input.shape)

        assert context.all_binding_shapes_specified

        host_outputs = []
        device_outputs = []
        for binding_index in output_binding_idxs:
            output_shape = context.get_binding_shape(binding_index)
            # Allocate buffers to hold output results after copying back to host
            buffer = np.empty(output_shape, dtype=np.float32)
            host_outputs.append(buffer)
            # Allocate output buffers on device
            device_outputs.append(cuda.mem_alloc(buffer.nbytes))

        return host_outputs, device_outputs

    # 固定size
    def get_fix_binding_shapes(
            self,
            context,
            input_binding_idxs,
            output_binding_idxs,
    ):
        '''
        回去固定的输入输出size，方便提前申请cuda空间
        Args:
            context:
            input_binding_idxs:
            output_binding_idxs:

        Returns:

        '''
        assert context.all_binding_shapes_specified

        fix_host_outputs = []
        fix_device_outputs = []
        fix_device_inputs = []

        for binding_index in input_binding_idxs:
            input_shape = context.get_binding_shape(binding_index)
            # Allocate buffers to hold output results after copying back to host
            buffer = np.empty(input_shape, dtype=np.float32)
            fix_device_inputs.append(cuda.mem_alloc(buffer.nbytes))

        for binding_index in output_binding_idxs:
            output_shape = context.get_binding_shape(binding_index)
            # Allocate buffers to hold output results after copying back to host
            buffer = np.empty(output_shape, dtype=np.float32)
            fix_host_outputs.append(buffer)
            # Allocate output buffers on device
            fix_device_outputs.append(cuda.mem_alloc(buffer.nbytes))

        return fix_host_outputs, fix_device_outputs, fix_device_inputs

    def judge_dynamic(self, shape):
        """ 判断是否动态
        :param shape:
        :return:
        """
        return any(dim is None or dim < 0 for dim in shape)

    def print_inputs_info(self, engine, context, input_binding_idxs):
        '''
        print input info and judge input is dynamic or fix
        Args:
            engine: trt engine
            context: engine context
            input_binding_idxs:

        Returns:

        '''

        for binding_index in input_binding_idxs:
            # If input shape is fixed, we'll just use it
            input_shape = context.get_binding_shape(binding_index)
            input_name = engine.get_binding_name(binding_index)
            print("\tInput [{}] shape: {}".format(input_name, input_shape))
            # If input shape is dynamic, we'll arbitrarily select one of the
            # the min/opt/max shapes from our optimization profile
            if self.judge_dynamic(input_shape):
                self.is_dynamic = True
                profile_index = context.active_optimization_profile
                profile_shapes = engine.get_profile_shape(profile_index, binding_index)
                print("\tProfile Shapes for [{}]: [kMIN {} | kOPT {} | kMAX {}]".format(input_name, *profile_shapes))
                print("\tInput [{}] shape was dynamic, setting inference shape to {}".format(input_name, input_shape))

    def print_stream_info(self, host_inputs, host_outputs):
        '''print context stream 的一些信息
        Args:
            host_inputs: 本地输入
            host_outputs: 本地输出

        Returns:
        '''
        print("Input Metadata")
        print("\tNumber of Inputs: {}".format(len(self.input_binding_idxs)))
        print(
            "\tInput Bindings for Profile {}: {}".format(self.context.active_optimization_profile,
                                                         self.input_binding_idxs))
        print("\tInput names: {}".format(self.input_names))
        print("\tInput shapes: {}".format([inp.shape for inp in host_inputs]))
        print("Output Metadata")
        print("\tNumber of Outputs: {}".format(len(self.output_binding_idxs)))
        print("\tOutput names: {}".format(self.output_names))
        print("\tOutput shapes: {}".format([out.shape for out in host_outputs]))
        print("\tOutput Bindings for Profile {}: {}".format(self.context.active_optimization_profile,
                                                            self.output_binding_idxs))

    def run_dynamic(self, image):
        # self.cfx.push()
        host_inputs = [image]
        # Allocate device memory for inputs. This can be easily re-used if the
        device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]
        # 动态推理时每次都需要根据input_data 申请空间
        host_outputs, device_outputs = self.setup_binding_shapes(self.context, host_inputs, self.input_binding_idxs,
                                                                 self.output_binding_idxs)

        # Copy host inputs to device, this needs to be done for each new input， 由host拷贝到device
        for h_input, d_input in zip(host_inputs, device_inputs):
            # cuda.memcpy_htod(d_input, h_input)
            cuda.memcpy_htod_async(d_input, h_input, self.stream)

        # Bindings are a list of device pointers for inputs and outputs
        bindings = device_inputs + device_outputs  # list的合并
        # Inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Copy outputs back to host to view results 将输出由gpu拷贝到cpu。
        output_data = []

        for h_output, d_output in zip(host_outputs, device_outputs):
            # cuda.memcpy_dtoh(h_output, d_output)
            cuda.memcpy_dtoh_async(h_output, d_output, self.stream)
            output_data.append(h_output.reshape(h_output.shape))

        self.stream.synchronize()
        # self.cfx.pop()

        return output_data

    def run_fix(self, image):
        ''' 静态输入下的推理

        Args:
            image:
        Returns:
        '''
        # self.cfx.push()

        host_inputs = [image]
        # Copy host inputs to device, this needs to be done for each new input(由host拷贝到device)
        for h_input, d_input in zip(host_inputs, self.fix_device_inputs):
            cuda.memcpy_htod_async(d_input, h_input, self.stream)

        # Bindings are a list of device pointers for inputs and outputs
        bindings = self.fix_device_inputs + self.fix_device_outputs

        # Inference
        self.context.execute_async(bindings=bindings, stream_handle=self.stream.handle)

        # Copy outputs back to host to view results（将输出由gpu拷贝到cpu）
        output_data = []
        for h_output, d_output in zip(self.fix_host_outputs, self.fix_device_outputs):
            cuda.memcpy_dtoh_async(h_output, d_output, self.stream)
            output_data.append(h_output.reshape(h_output.shape))

        self.stream.synchronize()
        # self.cfx.pop()

        return output_data

    def run(self, image):
        ''' trt推理，区分静态输入和动态输入

        Args:
            image:
        Returns:

        '''
        if self.is_dynamic:
            return self.run_dynamic(image)
        else:
            return self.run_fix(image)
