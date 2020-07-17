#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 9:37
# @Author  : TheTao
# @Site    : 
# @File    : params_utils.py
# @Software: PyCharm
import os
import pathlib
import tensorflow as tf

# tf消除警告
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# 获取工程的根目录, 这样路径就可以不区分操作系统
root = pathlib.Path(os.path.abspath(__file__)).parent
resultpath = os.path.join(root, "result")
log_dir = os.path.join(root, "log")
ckptpath = os.path.join(root, "check_point", "ckpt_2lstm128_nodrop")
embpath = os.path.join(root, "datas", "emb_file", "vec.txt")
trainpath = os.path.join(root, "datas", "prepare_data", "train.pkl")
devpath = os.path.join(root, "datas", "prepare_data", "dev.pkl")
testpath = os.path.join(root, "datas", "prepare_data", "test.pkl")
dictpath = os.path.join(root, "datas", "prepare_data", "dict.pkl")
train_logpath = os.path.join(root, "log", "train.log")
test_logpath = os.path.join(root, "log", "test.log")
sent_tagpath = os.path.join(root, "datas", "sentence", "train_sentence.pkl")


# 参数管理函数
def get_params():
    flags = tf.app.flags
    # 运行模式开关
    flags.DEFINE_boolean("clean", False, "clean train folder for new step train")
    flags.DEFINE_boolean("train", False, "whether train the model")
    flags.DEFINE_boolean("server", False, "if not run server on flask")
    # 特征映射参数
    flags.DEFINE_integer("char_dim", 100, "embedding size for characters")
    flags.DEFINE_integer("bound_dim", 20, "embedding size for boundary, 0 if not used")
    flags.DEFINE_integer("flag_dim", 50, "embedding size for char flag, 0 if not used")
    flags.DEFINE_integer("radical_dim", 50, "embedding size for radical, 0 if not used")
    flags.DEFINE_integer("pinyin_dim", 50, "embedding size for pinyin, 0 if not used")
    # 网络有关参数
    flags.DEFINE_float("clip", 5, "gradient clip")
    flags.DEFINE_float("dropout", 0.8, "dropout rate")
    flags.DEFINE_integer("batch_size", 16, "batch size")
    flags.DEFINE_integer("test_batch_size", 100, "test batch size")
    flags.DEFINE_float("lr", 1e-3, "initial learning rate")
    flags.DEFINE_integer("lstm_dim", 128, "num of hidden units in LSTM")
    # 是否使用预训练模型
    flags.DEFINE_boolean("pre_emb", True, "wither use pre-trained embedding")
    # 训练周期
    flags.DEFINE_integer("max_epoch", 10, "maximum training epochs")
    # 最大保存步数
    flags.DEFINE_integer("steps_check", 2, "steps per checkpoint")
    # 一些文件路径
    flags.DEFINE_string("ckpt_path", ckptpath, "Path to save model")
    flags.DEFINE_string("emb_file", embpath, "Path for pre_trained embedding")
    flags.DEFINE_string("train_file", trainpath, "Path for train data")
    flags.DEFINE_string("dev_file", devpath, "Path for dev data")
    flags.DEFINE_string("test_file", testpath, "Path for test data")
    flags.DEFINE_string("dict_file", dictpath, "Path for dict data")
    flags.DEFINE_string("result_path", resultpath, "Path for predict to file")
    flags.DEFINE_string("log_dir", log_dir, "Path for predict to log")
    flags.DEFINE_string("train_log_file", train_logpath, "File for train_log")
    flags.DEFINE_string("test_log_file", test_logpath, "File for test_log")
    flags.DEFINE_string("sent_tag_file", sent_tagpath, "File for sent_tag")
    # 再初始化返回
    FLAGS = tf.app.flags.FLAGS
    return FLAGS


if __name__ == '__main__':
    params = get_params()
    print(params.dict_file)
