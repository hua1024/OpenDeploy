## ResNet-Deploy

>The Pytorch implementation is [open-mmlab/mmclassification](https://github.com/open-mmlab/mmclassification).

support '0.12.0'


### [TensorRT](./TensorRT)

##### ResNet50

|Model |size |acc | Speed 2080Ti<br>(ms/FPS) | Params<br>(M) |FLOPs<br>(B)|Model Size<br>(M)|
|------|:---:| :---:|:---:|:---:|:---:|:---:|
|ResNet50-torch    |224  |95.86%      | 6.4/155 |23.51 |4.66  |89.2M  |  

|Model |size |acc | Speed 2080Ti<br>(ms/FPS) |
|------|:---:| :---:|:---:|
|ResNet50-torch    |224  |95.86%      | 6.4/155 |
|ResNet50-trt-python  |224  |95.86%      | 3.9/254    |
|ResNet50-trt-python-float16   |224  |95.86%      |2.5/399|
|ResNet50-trt-python-int8   |224  |51.2      | 17.3 |
|ResNet50-trt-cpp  |224  |95.86%       | 17.3 |
|ResNet50-trt-cpp-float16  |224  |51.2      | 17.3 |
|ResNet50-trt-cpp-int8  |224  |51.2      | 17.3 |

|Model |size |acc | Speed 2080Ti<br>(ms/FPS) |
|------|:---:| :---:|:---:|
|ResNet50-trt-python-bs4  |224  |95.86%      | 2.8/349|
|ResNet50-trt-python-bs8   |224  |95.86%      |2.2/442|
|ResNet50-trt-python-bs16   |224  |95.86%      | 2.1/461 |




### [NCNN](./NCNN)

|Model |size |acc | Speed 2080Ti<br>(ms) | Params<br>(M) |FLOPs<br>(B)|
| ------        |:---: | :---:       |:---:     |:---:  | :---: |
|ResNet50-torch  |224  |39.6      |9.8     |9.0 | 26.8 | 
|ResNet50-ncnn  |224  |39.6      |9.8     |9.0 | 26.8 | 

