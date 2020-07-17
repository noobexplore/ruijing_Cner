#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/11 17:24
# @Author  : TheTao
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import os
import time
import numpy as np
import tensorflow as tf
import itertools
from model import Model
from params_utils import get_params
from data_utils import BatchManager, get_dict, make_path, get_logger, get_sent_tag, augment_with_pretrained, \
    load_word2vec
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


def creat_model(session, model_class, ckpt_path, load_vec, param, id_to_char, logger, map_all):
    model = model_class(param, map_all)
    # 加载模型
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    # 看是否存在训练好的模型
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        # 如果存在就进行重新加载
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        # 此步骤非常重要，不初始化的话就无法读取到权重
        session.run(tf.global_variables_initializer())
        # 读取预训练模型
        if param.emb_file:
            # 先取得随机初始化的权重
            emb_weights = session.run(model.char_lookup.read_value())
            # 然后再加载预训练好的词向量
            emb_weights = load_vec(param.emb_file, id_to_char, param.char_dim, emb_weights)
            # 进行分配，后面训练的时候还是会修改
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model


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
    # 读取senc_tag为后续加载词向量做准备
    senc_tag = get_sent_tag(param.sent_tag_file)
    # 加载预训练向量
    dico_chars, char_to_id, id_to_char = augment_with_pretrained(
        mapping_dict['word'][2].copy(),
        param.emb_file,
        list(itertools.chain.from_iterable([[w[0] for w in s] for s in senc_tag])))
    # 获取总的训练集数据数量
    steps_per_epoch = train_manager.len_data
    # 配置GPU参数
    gpu_config = tf.ConfigProto()
    with tf.Session(config=gpu_config) as sess:
        # 初始化模型
        model = creat_model(sess, Model, param.ckpt_path, load_word2vec, param, id_to_char, logger,
                            map_all=mapping_dict)
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
                    logger.info("epoch:{}, step:{}/{}, avg_loss:{:>9.4f}".format(i + 1,
                                                                                 step % steps_per_epoch,
                                                                                 steps_per_epoch,
                                                                                 np.mean(loss)))
            # 保存模型
            model.save_model(sess, logger, i)
            logger.info('Epoch {}, total Loss {:.4f}'.format(i + 1, total_loss / train_manager.len_data))
            logger.info('Time taken for one epoch {:.4f} min, take {:.2f} h for rest of epoch\n'.format(
                (time.time() - start) / 60,
                ((param.max_epoch - i + 1) * (time.time() - start)) / 3600
            ))


if __name__ == '__main__':
    params = get_params()
    train(params)
