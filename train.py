#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/11 17:24
# @Author  : TheTao
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import time
import numpy as np
import tensorflow as tf
from data_utils import BatchManager, get_dict, make_path, get_logger
from model import Model
from params_utils import get_params
import warnings

warnings.filterwarnings("ignore")

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


def train(param):
    # 检查参数
    assert param.clip < 5.1, "gradient clip should't be too much"
    assert 0 <= param.dropout < 1, "dropout rate between 0 and 1"
    assert param.lr > 0, "learning rate must larger than zero"
    # 数据准备
    train_manager = BatchManager(param.batch_size, name='train')
    number_dataset = train_manager.len_data
    print("total of number train data is {}".format(number_dataset))
    # 创建相应的文件夹
    make_path(param)
    # 配置日志
    logger = get_logger(param.train_log_file)
    # 读取字典
    mapping_dict = get_dict(param.dict_file)
    # 搭建模型
    model = Model(param, mapping_dict)
    # 初始化参数
    init = tf.global_variables_initializer()
    # 获取总的训练集数据数量
    steps_per_epoch = train_manager.len_data
    # 配置GPU参数
    gpu_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    with tf.Session(config=gpu_config) as sess:
        sess.run(init)
        for i in range(param.max_epoch):
            loss = []
            total_loss = 0
            # 初始化时间
            start = time.time()
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, batch)
                # 这里计算平均loss
                loss.append(batch_loss)
                # 这里计算总的loss后面计算全部平均
                total_loss += batch_loss
                if step % 5 == 0:
                    logger.info("epoch:{}, step:{}/{}, avg_loss:{:>9.6f}".format(i + 1,
                                                                                 step % steps_per_epoch,
                                                                                 steps_per_epoch,
                                                                                 np.mean(loss)))
            # 保存模型
            model.save_model(sess, logger, i)
            logger.info('Epoch {}, total Loss {:.4f}'.format(i + 1, total_loss / train_manager.len_data))
            logger.info('Time taken for one epoch {:.4f} min, take {} h for rest of epoch\n'.format(
                (time.time() - start) / 60,
                ((param.max_epoch - i + 1) * (time.time() - start)) / 3600
            ))


if __name__ == '__main__':
    params = get_params()
    train(params)
