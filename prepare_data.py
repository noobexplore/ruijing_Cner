#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/2 17:00
# @Author  : TheTao
# @Site    : 
# @File    : prepare_data.py
# @Software: PyCharm
import os
import tqdm
import pandas as pd
from collections import Counter
from data_process import split_text
import jieba.posseg as psg
from cnradical import Radical

train_dir = 'ruijin_round1_train2_20181022'


def process_text(idx, split_method=None):
    """
    读取文本，切割，然后打上标记并提取词边界、词性、偏旁部首、拼音等文本特征
    :param idx:文件的名字 不含扩展名
    :param split_method:切割文本的方法
    :return:
    """
    data = {}
    # 获取句子
    if split_method is None:
        with open(f'./datas/{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        with open(f'./datas/{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = split_text(texts)
    data['word'] = texts
    # 获取标签，先全部打上O
    tag_list = ['O' for s in texts for x in s]
    # 读取对应的ann文件
    tag = pd.read_csv(f'./datas/{train_dir}/{idx}.ann', header=None, sep='\t')
    for i in range(tag.shape[0]):
        tag_item = tag.iloc[i][1].split(' ')
        # 开始打标签
        cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])
        tag_list[start] = 'B-' + cls
        for j in range(start + 1, end):
            tag_list[j] = 'I-' + cls
    # 做检查长度是否相等
    assert len([x for s in texts for x in s]) == len(tag_list)

    # 提取词性和词边界特征
    word_bounds = ['M' for item in tag_list]  # 保存每个词的边界
    word_flags = []  # 保存词性
    for text in texts:
        # 遍历带词性的切分
        for word, flag in psg.cut(text):
            # 单个词的时候
            if len(word) == 1:
                start = len(word_flags)
                word_bounds[start] = 'S'
                word_flags.append(flag)
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'
                word_flags += [flag] * len(word)
                # 这里end需要-1
                end = len(word_flags) - 1
                word_bounds[end] = 'E'
    # 这里保存词性，统一截断
    bounds = []
    flags = []
    tags = []
    start = 0
    end = 0
    for s in texts:
        ldx = len(s)
        end += ldx
        # 分句子显示
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        tags.append(tag_list[start:end])
        start += ldx
    data['bound'] = bounds
    data['flag'] = flags
    data['label'] = tags
    # 获取拼音特征

    return texts[0], tags[0], bounds[0], flags[0]


if __name__ == '__main__':
    print(process_text('0', split_method=split_text))
