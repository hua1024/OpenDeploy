### Install
Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
tensorrt==7.1.3.4


#### step 1: Prepare serialized engine file
    Follow the trt python demo readme to conver and save the serialized engine file.


#### step 2: Build the demo
```shell script
mkdir build
cd build
cmake ..
make -j4
```
#### step 3: Run
```shell script
./resnet ../weights/f16-r50_simpler.engine ../imgs/1.jpg
