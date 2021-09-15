# coding=utf-8  
# @Time   : 2021/3/26 10:34
# @Auto   : zzf-jeff


class GlobalSetting():
    label_path = 'weights/labels.txt'
    model_path = 'weights/fix_f16_r50-sim.engine'
    # model_path = 'weights/r50-sim.onnx'
    output_path = './output'
    img_path = 'imgs'

    img_size = (224, 224) # w,h
    infer_batch_size = 1
    conf_thresh = 0.3
    # 为了方便移动端,预处理img/255.
    mean = None
    std = None


opt = GlobalSetting()
