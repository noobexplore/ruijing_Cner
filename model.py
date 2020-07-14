#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 17:37
# @Author  : TheTao
# @Site    : 
# @File    : model.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode


def network(inputs, shapes, num_tags, lstm_dim=100, initializer=tf.truncated_normal_initializer()):
    """
    接受一个批次样本的特征数据，计算出网络的输出值
    :param num_tags:标签数量
    :param lstm_dim: LSTM维度
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
    # 取出真实长度作为时间序列
    num_time = tf.shape(inputs[keys[0]])[1]
    # 构建循环神经网络
    with tf.variable_scope('BiLSTM_layer1'):
        lstm_cell = {}
        for name in ['forward', 'backward']:
            # 实例化
            lstm_cell[name] = rnn.BasicLSTMCell(
                num_units=lstm_dim
            )
        # 将示例化的东西跑一下
        output1, final_states = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward'],
            lstm_cell['backward'],
            embed,
            dtype=tf.float32,
            sequence_length=lengths
        )
    # 这里需要将前后向进行拼接，在最后一个维度进行拼接 b,L,2*lstm_dim
    output1 = tf.concat(output1, axis=-1)
    # 第二层
    with tf.variable_scope('BiLSTM_layer2'):
        lstm_cell = {}
        for name in ['forward', 'backward']:
            # 实例化
            lstm_cell[name] = rnn.BasicLSTMCell(
                num_units=lstm_dim
            )
        # 将示例化的东西跑一下
        output2, final_states = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward'],
            lstm_cell['backward'],
            output1,
            dtype=tf.float32,
            sequence_length=lengths
        )
    output = tf.concat(output2, axis=-1)
    # 输出映射，合并为二维矩阵
    output = tf.reshape(output, [-1, 2 * lstm_dim])  # reshape成二维矩阵 [batch*maxlength, 2*lstmdim]
    with tf.variable_scope('project_layer1'):
        w = tf.get_variable(
            name='w',
            shape=[2 * lstm_dim, lstm_dim],
            initializer=initializer
        )
        b = tf.get_variable(
            name='b',
            shape=[lstm_dim],
            initializer=tf.zeros_initializer
        )
        # 映射
        output = tf.nn.relu(tf.matmul(output, w) + b)
    with tf.variable_scope('project_layer2'):
        w = tf.get_variable(
            name='w',
            shape=[lstm_dim, num_tags],
            initializer=initializer
        )
        b = tf.get_variable(
            name='b',
            shape=[num_tags],
            initializer=tf.zeros_initializer
        )
        # 映射最后一层不激活
        output = tf.matmul(output, w) + b
    output = tf.reshape(output, [-1, num_time, num_tags])
    return output, lengths  # batch_size, max_lenthg, num_tags


class Model(object):
    def __init__(self, dict_all, lr=0.0001):
        # 用到的参数值
        self.num_char = len(dict_all['word'][0])
        self.num_bound = len(dict_all['bound'][0])
        self.num_flag = len(dict_all['flag'][0])
        self.num_radical = len(dict_all['radical'][0])
        self.num_pinyin = len(dict_all['pinyin'][0])
        self.num_tags = len(dict_all['label'][0])
        self.char_dim = 100
        self.bound_dim = 20
        self.flag_dim = 50
        self.radical_dim = 50
        self.pinyin_dim = 50
        self.lstm_dim = 100
        self.lr = lr
        self.map = dict_all
        # 定义接受数据的placeholer
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='char_inputs')
        self.bound_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='bound_inputs')
        self.flag_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='flag_inputs')
        self.radical_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='radical_inputs')
        self.pinyin_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pinyin_inputs')
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
        # 不需要训练用来计数
        self.global_step = tf.Variable(0, trainable=False)
        # 计算模型输出值
        self.logits, self.lengths = self.get_logits(
            self.char_inputs,
            self.bound_inputs,
            self.flag_inputs,
            self.radical_inputs,
            self.pinyin_inputs
        )
        # 计算损失
        self.cost = self.get_loss(self.logits, self.targets, self.lengths)
        # 优化器优化，梯度截断技术
        with tf.variable_scope('optimizer'):
            opt = tf.train.AdamOptimizer(self.lr)
            # 计算出所有参数的导数
            grad_vars = opt.compute_gradients(self.cost)
            # 得到截断后的梯度
            clip_grad_vars = [[tf.clip_by_value(g, -5, 5), v] for g, v in grad_vars]
            # 使用截断后的梯度，对参数进行更新
            self.train_op = opt.apply_gradients(clip_grad_vars, self.global_step)
        # 只保留最近5次
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def get_logits(self, char, bound, flag, radical, pinyin):
        """
        接受一个批次样本的特征数据，计算出网络的输出值
        :param char: type of int, id of chars a tensor of shape 2-D [None, None]
        :param bound:id of chars a tensor of shape 2-D [None, None] with type of int
        :param flag:id of chars a tensor of shape 2-D [None, None] with type of int
        :param radical:id of chars a tensor of shape 2-D [None, None] with type of int
        :param pinyin:id of chars a tensor of shape 2-D [None, None] with type of int
        :return:3-d tensor [batch_size, max_length, num_tags]
        """
        shapes = {'char': [self.num_char, self.char_dim], 'bound': [self.num_bound, self.bound_dim],
                  'flag': [self.num_flag, self.flag_dim], 'radical': [self.num_radical, self.radical_dim],
                  'pinyin': [self.num_pinyin, self.pinyin_dim]}
        inputs = {'char': char, 'bound': bound, 'flag': flag, 'radical': radical, 'pinyin': pinyin}
        return network(inputs, shapes, lstm_dim=self.lstm_dim, num_tags=self.num_tags)

    def get_loss(self, output, targets, lengths):
        b = tf.shape(lengths)[0]
        num_steps = tf.shape(output)[1]
        with tf.variable_scope('crf_loss'):
            # 填充转移矩阵
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[b, 1, self.num_tags]), tf.zeros(shape=[b, 1, 1])],
                axis=-1
            )
            # pad不计算
            pad_logits = tf.cast(small * tf.ones([b, num_steps, 1]), tf.float32)
            logits = tf.concat([output, pad_logits], axis=-1)
            # 第二个维度进行拼接
            logits = tf.concat([start_logits, logits], axis=1)
            # 标签也要拼接
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([b, 1]), tf.int32), targets],
                axis=-1
            )
            self.trans = tf.get_variable(
                name='trans',
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=tf.truncated_normal_initializer()
            )
            # 计算LOSS，在传统的CRF中的logits是根据统计学去统计出来的分值
            log_likehood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths
            )
            return tf.reduce_mean(-log_likehood)

    def run_step(self, sess, batch, is_train=True):
        if is_train:
            feed_dict = {
                self.char_inputs: batch[0],
                self.bound_inputs: batch[1],
                self.flag_inputs: batch[2],
                self.targets: batch[3],
                self.radical_inputs: batch[4],
                self.pinyin_inputs: batch[5]
            }
            _, loss = sess.run([self.train_op, self.cost], feed_dict=feed_dict)
            return loss
        else:
            # 就不需要targets了
            feed_dict = {
                self.char_inputs: batch[0],
                self.bound_inputs: batch[1],
                self.flag_inputs: batch[2],
                self.radical_inputs: batch[4],
                self.pinyin_inputs: batch[5]
            }
        logits, lengths = sess.run([self.logits, self.lengths], feed_dict=feed_dict)
        return logits, lengths

    def decode(self, logtis, lengths, matrix):
        paths = []
        small = -1000
        start = np.asarray([[small * self.num_tags] + [0]])
        for score, length in zip(logtis, lengths):
            # 只取有效长度
            score = score[:length]
            pad = small * np.ones([length, 1])
            logtis = np.concatenate([score, pad], axis=-1)
            logtis = np.concatenate([start, logtis], axis=0)
            path, _ = viterbi_decode(logtis, matrix)
            # 去掉start
            paths.append(path[1:])
        return paths

    def predict(self, sess, batch):
        results = []
        chars = batch[0]
        # 先拿到转移矩阵
        matrix = self.trans.eval()
        logtis, lengths = self.run_step(sess, batch, is_train=False)
        # 获取预测的ID
        paths = self.decode(logtis, lengths, matrix)
        for i in range(len(paths)):
            length = lengths[i]
            # 第i句话的真实数据
            string = [self.map['word'][0][index] for index in chars[i][:length]]
            tags = [self.map['label'][0][index] for index in paths[i]]
            result = [k for k in zip(string, tags)]
            results.append(result)
        return results



