#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 15:47
# @Author  : TheTao
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import time
from model import Model
import tensorflow as tf
from params_utils import get_params
from data_utils import BatchManager, get_logger, get_dict, test_ner
import warnings

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 批量评估函数
def evaluate(sess, param, model, name, batchmanager, logger):
    # 拿到对应的一个批次测试结果集
    ner_results = model.evaluate(sess, batchmanager)
    # 预测结果保存到结果集
    eval_lines = test_ner(ner_results, param.result_path)
    # 这里是打印报告结果
    for line in eval_lines:
        logger.info(line)
    # 这里就拿到F1值
    f1 = float(eval_lines[1].strip().split()[-1])
    # 这里返回最佳的F1值
    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))


def test(param):
    # 检查参数
    assert param.clip < 5.1, "gradient clip should't be too much"
    assert 0 <= param.dropout < 1, "dropout rate between 0 and 1"
    assert param.lr > 0, "learning rate must larger than zero"
    # 获取batch_manager
    test_manager = BatchManager(param.test_batch_size, name='test')
    number_dataset = test_manager.len_data
    print("total of number test data is {}".format(number_dataset))
    # 配置日志
    logger = get_logger(param.test_log_file)
    # 读取字典
    mapping_dict = get_dict(param.dict_file)
    # 搭建模型
    model = Model(param, mapping_dict)
    # 配置GPU参数
    gpu_config = tf.ConfigProto()
    with tf.Session(config=gpu_config) as sess:
        logger.info("start testing...")
        start = time.time()
        # 首先检查模型是否存在
        ckpt_path = param.ckpt_path
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        # 看是否存在训练好的模型
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logger.info("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
            # 如果存在就进行重新加载
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logger.info("Cannot find the ckpt files!")
        # 开始评估
        evaluate(sess, param, model, "test", test_manager, logger)
        logger.info("The best_f1 on test_dataset is {:.2f}".format(model.best_test_f1.eval()))
        logger.info('Time test for {:.2f} batch is {:.2f} sec\n'.format(param.test_batch_size, time.time() - start))


if __name__ == '__main__':
    params = get_params()
    test(params)
