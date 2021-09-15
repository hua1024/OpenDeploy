# coding=utf-8  
# @Time   : 2020/12/29 18:32
# @Auto   : zzf-jeff

from argparse import ArgumentParser

import os
from tqdm import tqdm

CLASSES = [
    'danger', 'huge', 'small'
]


def load_label_data(txt_path):
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

    return img_list


def load_pred_data(txt_path):
    '''读取val.txt

    return eg: [(1.jpg,0)]
    '''
    img_list = []
    with open(txt_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip('\n')
            file, label, score = line.split(' ')
            img_list.append((file, label, score))

    return img_list


def main():
    parser = ArgumentParser()
    parser.add_argument('--label_txt_path', help='Image file')
    parser.add_argument('--pred_txt_path', help='Config file')

    args = parser.parse_args()

    label_info_list = load_label_data(args.label_txt_path)
    pred_info_list = load_pred_data(args.pred_txt_path)

    # 保证file 是对应的

    num_danger, num_huge, num_small = 0, 0, 0
    pred_danger_num, pred_huge_num, pred_small_num = 0, 0, 0
    for label_info, pred_info in zip(label_info_list, pred_info_list):
        label_idx = int(label_info[1])
        pred_idx = int(pred_info[1])

        if label_idx == 0:
            num_danger += 1
            if label_idx == pred_idx:
                pred_danger_num += 1

        elif label_idx == 1:
            num_huge += 1
            if label_idx == pred_idx:
                pred_huge_num += 1

        elif label_idx == 2:
            num_small += 1
            if label_idx == pred_idx:
                pred_small_num += 1

    print('danger acc: {}'.format(round((pred_danger_num / num_danger), 3)))
    print('huge acc: {}'.format(round((pred_huge_num / num_huge), 3)))
    print('small acc: {}'.format(round((pred_small_num / num_small), 3)))

if __name__ == '__main__':
    main()
