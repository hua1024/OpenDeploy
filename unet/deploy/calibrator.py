# coding=utf-8  
# @Time   : 2021/3/23 11:24
# @Auto   : zzf-jeff

import os
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# calibrator
# IInt8EntropyCalibrator2
# IInt8LegacyCalibrator
# IInt8EntropyCalibrator
# IInt8MinMaxCalibrator

class Calibrator(trt.IInt8EntropyCalibrator):

    def __init__(self, input_layers,stream, cache_file=""):
        trt.IInt8EntropyCalibrator.__init__(self)
        # data stream to qua

        self.stream = stream
        self.input_layers = input_layers
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self,  names):
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]
        # for i in self.input_layers[0]:
        #     assert names[0] != i
        #
        # bindings[0] = int(self.d_input)
        # return bindings

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            print("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)