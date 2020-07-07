#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 17:37
# @Author  : TheTao
# @Site    : 
# @File    : model.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


def network(char, bound, flag, radical, pinyin):
    """
    接受一个批次样本的特征数据，计算出网络的输出值
    :param char:id of chars a tensor of shape 2-D [None, None] with type of int
    :param bound:id of chars a tensor of shape 2-D [None, None] with type of int
    :param flag:id of chars a tensor of shape 2-D [None, None] with type of int
    :param radical:id of chars a tensor of shape 2-D [None, None] with type of int
    :param pinyin:id of chars a tensor of shape 2-D [None, None] with type of int
    :return:
    """
    # 特征嵌入，将所有特征的ID转换成一个固定长度的向量
    embedding = []
    with tf.variable_scope('char_embedding'):
        char_lookup = tf.get_variable(
            name='char_embedding',
            shape=[]
        )


class Model(object):
    def __init__(self, dict_all, ):
        self.num_char = len(dict_all['word'][0])
        self.num_bound = len(dict_all['bound'][0])
        self.num_flag = len(dict_all['flag'][0])
        self.num_radical = len(dict_all['radical'][0])
        self.num_pinyin = len(dict_all['pinyin'][0])
        self.num_entity = len(dict_all['label'][0])
        self.char_dim = 100
        self.bound_dim = 20
        self.flag_dim = 50
        self.radical_dim = 50
        self.pinyin_dim = 50

    def get_logits(self, char, bound, flag, radical, pinyin):
        """
        接受一个批次样本的特征数据，计算出网络的输出值
        :param char: type of int, id of chars a tensor of shape 2-D [None, None]
        :param bound:id of chars a tensor of shape 2-D [None, None] with type of int
        :param flag:id of chars a tensor of shape 2-D [None, None] with type of int
        :param radical:id of chars a tensor of shape 2-D [None, None] with type of int
        :param pinyin:id of chars a tensor of shape 2-D [None, None] with type of int
        :return:
        """
        return network(char, bound, flag, radical, pinyin)
