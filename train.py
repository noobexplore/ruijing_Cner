#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/11 17:24
# @Author  : TheTao
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import tensorflow as tf
from data_utils import BatchManager, get_dict
from model import Model

batch_size = 20
dict_file = './datas/prepare_data/dict.pkl'


def train():
    # 数据准备
    train_data = BatchManager(batch_size, name='train')
    # 读取字典
    mapping_dict = get_dict(dict_file)
    # 搭建模型
    model = Model(mapping_dict)


if __name__ == '__main__':
    train()
