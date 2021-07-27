# coding=utf-8  
# @Time   : 2021/3/6 10:32
# @Auto   : zzf-jeff

import sys, os
import argparse
import tensorrt as trt
sys.path.append('./')

from deploy_trt.calibrator import Calibrator
from deploy_trt.data_stream import CalibStream


def onnx2trt(
        onnx_file_path,
        engine_file_path,
        log_level='ERROR',
        max_batch_size=1,
        max_workspace_size=1,
        dynamic_shape=None,
        fp16_mode=False,
        strict_type_constraints=False,
        int8_mode=False,
        calibrator_stream=None,
        calibration_table_path=None,
        save_engine=False):
    """build TensorRT model from Onnx model.

    Args:
        onnx_file_path (string or io object): Onnx model name
        log_level (string, default is ERROR): TensorRT logger level, now
            INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        max_batch_size (int, default=1): The maximum batch size which can be
            used at execution time, and also the batch size for which the
            ICudaEngine will be optimized.
        max_workspace_size (int, default is 1): The maximum GPU temporary
            memory which the ICudaEngine can use at execution time. default is
            1GB.
        fp16_mode (bool, default is False): Whether or not 16-bit kernels are
            permitted. During engine build fp16 kernels will also be tried when
            this mode is enabled.
        strict_type_constraints (bool, default is False): When strict type
            constraints is set, TensorRT will choose the type constraints that
            conforms to type constraints. If the flag is not enabled higher
            precision implementation may be chosen if it results in higher
            performance.
        int8_mode (bool, default is False): Whether Int8 mode is used.
        int8_calibrator : calibrator for int8 mode, if None, default
            calibrator will be used as calibration data.
    """
    # init trt logger
    logger = trt.Logger(getattr(trt.Logger, log_level))

    ## 如果存在engine，则直接返回
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        ## 从序列化engine文件中创建推理引擎
        with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    # use logger builder trt-builder
    builder = trt.Builder(logger)
    # use trt-builder builder network
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # us parser set onnx model
    parser = trt.OnnxParser(network, logger)
    if isinstance(onnx_file_path, str):
        with open(onnx_file_path, 'rb') as f:
            flag = parser.parse(f.read())
    else:
        flag = parser.parse(onnx_file_path.read())
    if not flag:
        # get error
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    # re-order output tensor
    output_tensors = [network.get_output(i)
                      for i in range(network.num_outputs)]

    [network.unmark_output(tensor) for tensor in output_tensors]
    # rename out tensor
    for tensor in output_tensors:
        identity_out_tensor = network.add_identity(tensor).get_output(0)
        identity_out_tensor.name = 'identity_{}'.format(tensor.name)
        network.mark_output(tensor=identity_out_tensor)

    # set max_batch_size and config
    builder.max_batch_size = max_batch_size

    config = builder.create_builder_config()
    ## (1 << 25) 1G
    config.max_workspace_size = max_workspace_size * (1 << 30)

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    if int8_mode:
        # nt8量化
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(["input"], calibrator_stream, calibration_table_path)
        print('Int8 mode enabled')
    # set dynamic batch size profile
    profile = builder.create_optimization_profile()

    if dynamic_shape is not None:
        # set dynamic_shape use min,opt,max shape
        # such as : (1,3,224,224),(4,3,224,224),(16,3,224,224)
        profile.set_shape(network.get_input(0).name, *dynamic_shape)
    else:
        # num_inputs 一般都是one，实际上下面的用来做固定shape的配置
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            name = tensor.name
            shape = tensor.shape[1:]
            min_shape = (1,) + shape
            opt_shape = ((1 + max_batch_size) // 2,) + shape
            max_shape = (max_batch_size,) + shape
            # # 这里设置了min,opt,max三种shape的大小，需要注意
            profile.set_shape(name, min_shape, opt_shape, max_shape)
            # 动态设置参数，小于或者大于都会error在推理时

    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)

    if engine is None:
        print('Failed to create the engine')
        return None
    print("Completed creating the engine")

    if save_engine:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

    return engine


def main():
    args = parse_args()
    # get info
    onnx_input = args.onnx_input
    engine_output = args.engine_output
    fp16_mode = args.fp16_mode
    int8_mode = args.int8_mode
    max_batch_size = args.max_batch_size
    dynamic_shape = args.dynamic_shape
    # 如果是静态输入 dynamic_shape=None
    # 如果是动态输入 dynamic_shape=[min,opt,max]
    if dynamic_shape is not None:
        dynamic_shape = eval(dynamic_shape)
    if args.int8_mode:
        # int8 calibration
        batch_size = 16
        max_calibration_size = 500  # 校准集数量
        img_size = (3, 736, 1280)
        max_batches = max_calibration_size / batch_size
        calib_img_dir = ''
        calibration_stream = CalibStream(batch_size, img_size, max_batches, calib_img_dir)
        calibration_table_path = 'weights/5s_calibration.cache'
    else:
        calibration_stream = None
        calibration_table_path = None

    engine = onnx2trt(
        onnx_input,
        engine_output,
        max_batch_size=max_batch_size,
        fp16_mode=fp16_mode,
        int8_mode=int8_mode,
        dynamic_shape=dynamic_shape,
        calibrator_stream=calibration_stream,
        calibration_table_path=calibration_table_path,
        save_engine=True
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--onnx_input', type=str, help='train config file path')
    parser.add_argument('--engine_output', type=str, help='onnx save path')
    parser.add_argument('--fp16_mode', action='store_true', help='is float16')
    parser.add_argument('--int8_mode', action='store_true', help='is int8')
    parser.add_argument('--max_batch_size', type=int, default=4, help='max batch size')
    parser.add_argument('--dynamic_shape', type=str, default=None,
                        help='dynamic shape,min/opt/max shape')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
