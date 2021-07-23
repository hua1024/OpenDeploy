# coding=utf-8  
# @Time   : 2021/3/26 10:34
# @Auto   : zzf-jeff


class GlobalSetting():

    label_path = './coco.names'
    model_path = './weights/yolov5x-simpler.engine'
    # model_path = './weights/yolov5s.pt'
    output_path = './output'
    img_path = './test_imgs'

    conf_thresh = 0.3
    iou_thresh = 0.45

    anchors = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ]


opt = GlobalSetting()
