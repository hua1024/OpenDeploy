### 简介
>unet的tensorrt转换及推理工程
>
>训练工程 https://github.com/milesial/Pytorch-UNet

#### 1. 基础镜像环境
```shell script
# 61机器
# 镜像 : tensorrt-7.1.3:latest
# tensorrt-7.1.3、torch-1.6、torchvision-0.7.0
```

#### 2. torch2onnx
```shell script
# 将torch的模型转成onnx格式
python3 deploy/torch_to_onnx.py --weights weights/unet.pth --onnx_output weights/unet.onnx --is_dynamic --input_shape '1,3,300,300'
#注意 opset_version=12
# ****这里网络结构部分要修改，有部分坑
```
```python
# 上采样的forward
def forward(self, x1, x2):
    if self.training:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    else:
        # 这里由于trt上采样时Upsample支持不太好，改成interpolate，同时需要给固定尺寸，不能给缩放值
        # 同时trt动态推理时，当前不支持F.pad操作，目前改成了F.interpolate的形式，测试时没有什么影响的，后续有问题再想办法处理
        H, W = x1.size()[2], x1.size()[3]
        x1 = F.interpolate(x1, size=(H * 2, W * 2), mode='bilinear', align_corners=True)
        H1, W1 = x2.size()[2], x2.size()[3]
        x1 = F.interpolate(x1, size=(H1, W1), mode='bilinear', align_corners=True)
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

```


#### 3. onnxsim
```shell script
# 去除onnxsim中多余的op操作
# --dynamic-input-shape 如果输入的batch和size会发生改变，则需要加上
# input:1,3,640,640
python3 -m onnxsim weights/unet.onnx weights/unet-sim.onnx --input-shape input:1,3,500,500 --dynamic-input-shape
```


#### 4. onnx2trt
```shell script
python3 deploy/onnx_to_trt.py --onnx_input weights/unet-sim.onnx --engine_output weights/f16-unet-sim.engine --fp16_mode --max_batch_size 1 --dynamic_shape '[[1,3,10,10],[1,3,300,300],[1,3,500,500]]'
```