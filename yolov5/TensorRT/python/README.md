# YOLOV5-TensorRT in python

### Install


### Convert model

#### step 1: Convert torch model to onnx model
```shell script
# 将torch的模型转成onnx格式
# 默认保存在./weights/yolov5s.onnx
# 注意 opset_version=12
# 原作者: python3 models/export.py --weights ./weights/yolov5x.pt --img 640 --batch 1
```
#### step 2: Remove onnx model redundant op
```shell script
# 去除onnxsim中多余的op操作
# --dynamic-input-shape 如果输入的batch和size会发生改变，则需要加上
# input:1,3,640,640 -> 输入的节点名称 : bchw
python3 -m onnxsim weights/yolov5x.onnx weights/yolov5x-simpler.onnx --input-shape input:1,3,640,640 --dynamic-input-shape
```
#### step 3: Cpnvert onnx model to trt model
```shell script
# trt 动态输入
python3 deploy_trt/onnx_to_trt.py --onnx_input weights/yolov5x-simpler.onnx --engine_output weights/yolov5x-simpler.engine --max_batch_size 1 --dynamic_shape '[[1,3,128,128],[1,3,416,416],[1,3,640,640]]'
# 注意最小最大尺寸的限制，同时占用的显存是通过设置最大的显存来开辟的
```


### Demo

```shell script
# 参数都在config里面修改
python3 main.py
```