# ResNet-CPP-NCNN

Cpp base on [ncnn](https://github.com/Tencent/ncnn).

## Tutorial

### Step1
Clone [ncnn](https://github.com/Tencent/ncnn) first, then please following [build tutorial of ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build) to build on your own device.

### Step2
Use provided tools to generate onnx file.

### Step3
Generate ncnn param and bin file.
```shell
cd <path of ncnn>
cd build/tools/ncnn
./onnx2ncnn r50_simpler.onnx r50_simpler.param r50_simpler.bin
```

### Step4
Copy or Move resnet.cpp file into ncnn/examples, modify the CMakeList.txt, then build
```shell script
cp resnet.cpp ncnn/examples

# append ncnn_add_example(resnet)
vim ncnn/examples/CMakeLists.txt

make 
```

### Step5
Inference image
```shell
./resnet test.jpg r50_simpler.param r50_simpler.bin
```

