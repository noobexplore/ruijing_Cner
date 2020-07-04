#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/4 9:18
# @Author : TheTAO
# @Site : 
# @File : data_utils.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os


def get_data_with_windows(name='train'):
    with open(f'datas/prepare_data/dict.pkl', 'rb') as f:
        map_dict = pickle.load(f)

    def item2id(data, w2i):
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data]

    results = []
    root = os.path.join('datas/prepare_data', name)
    files = list(os.listdir(root))
    for file in tqdm(files):
        result = []
        path = os.path.join(root, file)
        samples = pd.read_csv(path, sep=',')
        num_samples = len(samples)
        # 先拿到分割下标
        sep_idx = [-1] + samples[samples['word'] == 'sep'].index.tolist() + [num_samples]
        # 获取所有句子进行ID的转化
        for i in range(len(sep_idx) - 1):
            start = sep_idx[i] + 1
            end = sep_idx[i + 1]
            id_data = []
            # 开始转换，拿到每个
            for feature in samples.columns:
                id_data.append(item2id(list(samples[feature])[start:end], map_dict[feature][1]))
            result.append(id_data)
        # 拼接长短句，数据增强
        two = []
        for i in range(len(result) - 1):
            first = result[i]
            second = result[i + 1]
            # 拼接两个
            two.append([first[k] + second[k] for k in range(len(first))])
        # 拼接三个
        three = []
        for i in range(len(result) - 2):
            first = result[i]
            second = result[i + 1]
            third = result[i + 2]
            # 拼接三个
            three.append([first[k] + second[k] + third[k] for k in range(len(first))])
        results.extend(result + two + three)
    return results


if __name__ == '__main__':
    get_data_with_windows()
