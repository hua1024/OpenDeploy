# coding=utf-8  
# @Time   : 2021/3/26 15:13
# @Auto   : zzf-jeff

import argparse
from tqdm import tqdm
from config import opt
from classifier import Classifier
from utils import *


def main():
    # get info
    model_path = opt.model_path
    img_path = opt.img_path
    output_path = opt.output_path
    label_path = opt.label_path
    img_size = opt.img_size

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_list = load_img_data_batch(img_path, opt.infer_batch_size)

    classifier = Classifier(
        model_path=model_path,
        label_path=label_path,
        img_size=img_size,
        conf_thresh=opt.conf_thresh,
        mean=opt.mean,
        std=opt.std,
    )

    # Run inference

    for idx, files in enumerate(tqdm(img_list)):
        inputs = [cv2.imread(file) for file in files]
        preds = classifier.run(inputs)
        for pred, file in zip(preds, files):
            basename = os.path.basename(file)
            label = '%s %.2f' % (pred["pred_label"], pred["pred_score"])
            with open(os.path.join(output_path, os.path.splitext(basename)[0] + '.txt'), 'w') as f:
                f.write(('%s' + '\n') % (label))  # label format
            img = cv2.imread(file)
            plot_classify_label((30, 30), img, label)
            cv2.imwrite(os.path.join(output_path, basename), img)


if __name__ == '__main__':
    main()
