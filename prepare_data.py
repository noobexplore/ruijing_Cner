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
        with open(f'../datas/{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:






