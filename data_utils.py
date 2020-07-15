#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/4 9:18
# @Author : TheTAO
# @Site : 
# @File : data_utils.py
# @Software: PyCharm
import os
import random
import math
import pickle
import logging
from tqdm import tqdm
import pandas as pd


def get_dict(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict


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
        # 去掉换行符
        if len(result[-1][0]) == 1:
            result = result[:-1]
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
    # 保存到文件, 这里将训练集分为train和dev
    if name == 'train':
        split_ratio = [0.8, 0.2]
        total = len(results)
        p1 = int(total * split_ratio[0])
        p2 = int(total * (split_ratio[0] + split_ratio[1]))
        with open(f'datas/prepare_data/train.pkl', 'wb') as f:
            pickle.dump(results[:p1], f)
        with open(f'datas/prepare_data/dev.pkl', 'wb') as f:
            pickle.dump(results[p1:p2], f)
    else:
        with open(f'datas/prepare_data/test.pkl', 'wb') as f:
            pickle.dump(results, f)


# batch管理对象
class BatchManager(object):
    def __init__(self, batch_size, name='train'):
        # 这里就直接读取文件了
        with open(f'datas/prepare_data/' + name + '.pkl', 'rb') as f:
            data = pickle.load(f)
        # 初始化排序和填充
        self.batch_data = self.sort_pad(data, batch_size)
        # 计算总的batch长度
        self.len_data = len(self.batch_data)

    def sort_pad(self, data, batch_size):
        # 计算总共有多少个批次
        num_batch = int(math.ceil(len(data) / batch_size))
        # 安装句子长度排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        # 获取batch
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * int(batch_size): (i + 1) * int(batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        chars, bounds, flags, radicals, pinyins, targets = [], [], [], [], [], []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            char, bound, flag, target, radical, pinyin = line
            # 需要填充的个数
            padding = [0] * (max_length - len(char))
            chars.append(char + padding)
            bounds.append(bound + padding)
            flags.append(flag + padding)
            targets.append(target + padding)
            radicals.append(radical + padding)
            pinyins.append(pinyin + padding)
        return [chars, bounds, flags, radicals, pinyins, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


# 获取日志文件
def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# 创建对应的文件夹
def make_path(param):
    # 预测结果集文件夹
    if not os.path.isdir(param.result_path):
        os.makedirs(param.result_path)
    # 模型保存文件夹
    if not os.path.isdir(param.ckpt_path):
        os.makedirs(param.ckpt_path)
    # 日志文件
    if not os.path.isdir(param.log_dir):
        os.makedirs(param.log_dir)


if __name__ == '__main__':
    get_data_with_windows('train')
    get_data_with_windows('test')
