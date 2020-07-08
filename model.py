#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 17:37
# @Author  : TheTao
# @Site    : 
# @File    : model.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


def network(inputs, shapes, lstm_dim=100, initializer=tf.truncated_normal_initializer):
    """
    接受一个批次样本的特征数据，计算出网络的输出值
    :param inputs: id of chars a tensor of shape 2-D [None, None] with type of int
    :param initializer:truncated_normal_initializer
    :param shapes:dim
    :return:
    """
    # 特征嵌入，将所有特征的ID转换成一个固定长度的向量
    embedding = []
    # 增加函数通用性
    keys = list(shapes.keys())
    for key in keys:
        with tf.variable_scope(key + '_embedding'):
            lookup = tf.get_variable(
                name=key + '_embedding',
                shape=shapes[key],
                initializer=initializer
            )
            # char映射操作
            embedding.append(tf.nn.embedding_lookup(lookup, inputs[key]))
    # 在最后一个维度上拼接shape [None, None, char_dim+...+pinyin_dim]
    embed = tf.concat(embedding, axis=-1)  # axis=-1为最后一个维度
    # 直接算出实际长度, sign就是正数为1负数为-1
    sign = tf.sign(tf.abs(inputs[keys[0]]))
    # 求出每个句子的真实长度
    lengths = tf.reduce_sum(sign, reduction_indices=1)
    # 构建循环神经网络
    with tf.variable_scope('BiLSTM_layer1'):
        lstm_cell = {}
        for name in ['forward1', 'backward1']:
            # 实例化
            lstm_cell[name] = rnn.BasicLSTMCell(
                num_units=lstm_dim,
                initializer=initializer
            )
        # 将示例化的东西跑一下
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward1'],
            lstm_cell['backward1'],
            embed,
            dtype=tf.float32,
            sequence_length=lengths
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
        self.lstm_dim = 100

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
        shapes = {'char': [self.num_char, self.char_dim], 'bound': [self.num_bound, self.bound_dim],
                  'flag': [self.num_flag, self.flag_dim], 'radical': [self.num_radical, self.radical_dim],
                  'pinyin': [self.num_pinyin, self.pinyin_dim]}
        inputs = {'char': char, 'bound': bound, 'flag': flag, 'radical': radical, 'pinyin': pinyin}
        return network(inputs, shapes, lstm_dim=self.lstm_dim)
