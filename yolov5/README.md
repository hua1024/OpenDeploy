## YOLOV5-Deploy

>The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

Currently, we support yolov5  v3.0, v3.1, v4.0.


### [TensorRT](./TensorRT)

##### YOLOV5x

|Model |size |mAP<sup>test<br>0.5:0.95 | Speed 2080Ti<br>(ms) | Params<br>(M) |FLOPs<br>(B)|
|------|:---:| :---:                   |:---:                 |:---:          | :---:      |
|YOLOV5x-torch    |640  |51.2      | 17.3 |99.1 |281.9  | 
|YOLOV5x-trt-cpp  |640  |51.2      | 17.3 |99.1 |281.9  |
|YOLOV5x-trt-cpp-float16  |640  |51.2      | 17.3 |99.1 |281.9  |
|YOLOV5x-trt-cpp-int8  |640  |51.2      | 17.3 |99.1 |281.9  |
|YOLOV5x-trt-python  |640  |39.6      |9.8     |9.0 | 26.8 | 
|YOLOV5x-trt-python-float16   |640  |51.2      | 17.3 |99.1 |281.9  |
|YOLOV5x-trt-python-int8   |640  |51.2      | 17.3 |99.1 |281.9  |


##### YOLOV5s


|Model |size |mAP<sup>test<br>0.5:0.95 | Speed 2080Ti<br>(ms) | Params<br>(M) |FLOPs<br>(B)|
| ------        |:---: | :---:       |:---:     |:---:  | :---: |
|YOLOV5s-torch  |640  |51.2      | 17.3 |99.1 |281.9  | 
|YOLOV5s-trt-cpp  |640  |51.2      | 17.3 |99.1 |281.9  |
|YOLOV5s-trt-cpp-float16  |640  |51.2      | 17.3 |99.1 |281.9  |
|YOLOV5s-trt-cpp-int8  |640  |51.2      | 17.3 |99.1 |281.9  |
|YOLOV5s-trt-python  |640  |39.6      |9.8     |9.0 | 26.8 | 
|YOLOV5s-trt-python-float16   |640  |51.2      | 17.3 |99.1 |281.9  |
|YOLOV5s-trt-python-int8   |640  |51.2      | 17.3 |99.1 |281.9  |



### [NCNN](./NCNN)

|Model |size |mAP<sup>test<br>0.5:0.95 | Speed 2080Ti<br>(ms) | Params<br>(M) |FLOPs<br>(B)|
| ------        |:---: | :---:       |:---:     |:---:  | :---: |
|YOLOV5s-torch  |640  |39.6      |9.8     |9.0 | 26.8 | 
|YOLOV5x-torch  |640  |51.2      | 17.3 |99.1 |281.9  | 
|YOLOV5s-ncnn  |640  |39.6      |9.8     |9.0 | 26.8 | 
|YOLOV5x-ncnn  |640  |51.2      | 17.3 |99.1 |281.9  |

