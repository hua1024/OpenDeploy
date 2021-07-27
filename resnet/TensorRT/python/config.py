# coding=utf-8  
# @Time   : 2021/3/26 10:34
# @Auto   : zzf-jeff


class GlobalSetting():
    label_path = 'labels.txt'
    model_path = 'weights/latest-simpler.engine'
    # model_path = './weights/yolov5s.pt'
    output_path = './output'
    img_path = 'test_imgs'

    conf_thresh = 0.3

    mean = None
    std = None

opt = GlobalSetting()
