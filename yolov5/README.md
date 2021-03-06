### 简介
>yolov5的tensorrt转换及推理工程
>

#### 1. environment
```shell script
# tensorrt-7.1.3、torch-1.6、torchvision-0.7.0
```

#### 2. torch2onnx
```shell script
# 将torch的模型转成onnx格式
# 默认保存在./weights/yolov5s.onnx
# 注意 opset_version=12
# 原作者: python3 models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
```


#### 3. onnxsim
```shell script
# 去除onnxsim中多余的op操作
# --dynamic-input-shape 如果输入的batch和size会发生改变，则需要加上
# input:1,3,640,640 -> 输入的节点名称 : bchw
python3 -m onnxsim weights/5x_best.onnx weights/5x_best-sim.onnx --input-shape input:1,3,640,640 --dynamic-input-shape
```


#### 4. onnx2trt
```shell script
# trt float16 动态输入
python3 deploy_trt/onnx_to_trt.py --onnx_input weights/5x_best-sim.onnx --engine_output weights/f16-5x_best-sim.engine --fp16_mode --max_batch_size 1 --dynamic_shape '[[1,3,128,128],[1,3,416,416],[1,3,640,640]]'

# 注意最小最大尺寸的限制，同时占用的显存是通过设置最大的显存来开辟的
```

