# coding=utf-8  
# @Time   : 2021/3/26 10:34
# @Auto   : zzf-jeff


class GlobalSetting():
    label_path = 'coco.names'
    onnx_path = 'weights/yolov5s-sim.onnx'
    engine_path = 'weights/f16-yolov5s-sim.engine'
    output_path = 'det_output'
    img_path = 'imgs'

    conf_thresh = 0.7
    iou_thresh = 0.6


opt = GlobalSetting()
