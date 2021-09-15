### Install
Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
tensorrt==7.1.3.4

pip install -r requirements.txt

### Convert model

#### step 1: Convert torch model to onnx model
```shell script
# opset_version=12
python3 tools/deployment/pytorch2onnx.py configs/resnet/car_r50.py --checkpoint work_dirs/car_r50/latest.pth --output-file work_dirs/car_r50/r50.onnx --show --dynamic-export
```
#### step 2: Remove onnx model redundant op
```shell script
# 去除onnxsim中多余的op操作
python3 -m onnxsim weights/r50.onnx weights/r50-sim.onnx --input-shape input:1,3,224,224 --dynamic-input-shape
```
#### step 3: Convert onnx model to trt model
```shell script
# trt 动态输入
python3 deploy_trt/onnx_to_trt.py --onnx_input weights/r50-sim.onnx --engine_output weights/r50_simpler.engine --max_batch_size 16 --dynamic_shape '[[1,3,224,224],[8,3,224,224],[16,3,224,224]]'
# 注意最小最大尺寸的限制，同时占用的显存是通过设置最大的显存来开辟的
```

### Demo

```shell script
# 参数都在config里面修改
python3 main.py
```