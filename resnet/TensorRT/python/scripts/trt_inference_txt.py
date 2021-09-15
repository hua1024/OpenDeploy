# coding=utf-8  
# @Time   : 2021/3/26 15:13
# @Auto   : zzf-jeff

# cal fps and test txt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append("./")
from config import opt
from classifier import Classifier
from utils import *

CLASSES = [
    'danger', 'huge', 'small'
]


def load_val_data(txt_path):
    '''读取val.txt

    return eg: [(1.jpg,0)]
    '''
    img_list = []
    with open(txt_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip('\n')
            file, label = line.split(' ')
            img_list.append((file, label))

    img_list = split_batch(img_list, batch_size=opt.infer_batch_size)

    return img_list


def main():
    parser = ArgumentParser()
    parser.add_argument('--txt_path', help='Image file')
    parser.add_argument('--out_txt', help='output file', default='scripts/trt_py_results.txt')
    args = parser.parse_args()

    # get infoa
    txt_path = args.txt_path
    output_txt = args.out_txt
    model_path = opt.model_path
    label_path = opt.label_path
    img_size = opt.img_size

    classifier = Classifier(
        model_path=model_path,
        label_path=label_path,
        img_size=img_size,
        conf_thresh=opt.conf_thresh,
        mean=opt.mean,
        std=opt.std,
    )

    info_list = load_val_data(txt_path)
    print("[INFO] Val info_list: ", len(info_list))

    all_time_list = []
    with open(output_txt, 'w', encoding='utf-8') as fw:
        for infos in tqdm(info_list):
            # (file, label) = infos
            root_path = "/home/pcl/zzf/san_code/car_clas_his/mmclassification"
            inputs = [cv2.imread(os.path.join(root_path, info[0])) for info in infos]
            # inference
            torch.cuda.synchronize()
            start_time = time.time()
            preds = classifier.run(inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            all_time_list.append(end_time - start_time)

            for pred, info in zip(preds, infos):
                (file, label) = info
                predict_class = CLASSES.index(pred['pred_label'])
                pred_score = round(pred['pred_score'], 5)
                line = "{} {} {}".format(file, predict_class, pred_score) + '\n'
                fw.write(line)

    print("FPS: %f,Mean time:%f (ms)" % (
        1.0 / (sum(all_time_list) / (len(info_list) * opt.infer_batch_size)),
        (sum(all_time_list) / (len(info_list) * opt.infer_batch_size)) * 1000))


if __name__ == '__main__':
    main()
