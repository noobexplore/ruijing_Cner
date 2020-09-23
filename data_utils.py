#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/4 9:18
# @Author : TheTAO
# @Site : 
# @File : data_utils.py
# @Software: PyCharm
import os
import random
import math
import pickle
import logging
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from conlleval import return_report
import codecs

entities_dict_chinese = {
    'Level': '等级-Level',
    'Test_Value': '检测值-Test_Value',
    'Test': '测试类-Test',
    'Anatomy': '解剖类-Anatomy',
    'Amount': '程度-Amount',
    'Disease': '疾病类-Disease',
    'Drug': '药物类-Drug',
    'Treatment': '治疗方法-Treatment',
    'Reason': '原因-Reason',
    'Method': '方法类-Method',
    'Duration': '持续时间-Duration',
    'Operation': '手术类-Operation',
    'Frequency': '频率-Frequency',
    'Symptom': '症状类-Symptom',
    'SideEff': '副作用-SideEff'
}


def get_dict(path):
    with open(path, 'rb') as f:
        char_dict = pickle.load(f)
    return char_dict


def get_sent_tag(path):
    with open(path, 'rb') as f:
        sent_tag = pickle.load(f)
    return sent_tag


def get_data_with_windows(name='train'):
    with open(f'datas/prepare_data/dict.pkl', 'rb') as f:
        map_dict = pickle.load(f)

    def item2id(data, w2i):
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data]

    results = []
    root = os.path.join('datas/prepare_data', name)
    files = list(os.listdir(root))
    for file in tqdm(files):
        result = []
        path = os.path.join(root, file)
        samples = pd.read_csv(path, sep=',')
        num_samples = len(samples)
        # 先拿到分割下标
        sep_idx = [-1] + samples[samples['word'] == 'sep'].index.tolist() + [num_samples]
        # 获取所有句子进行ID的转化
        for i in range(len(sep_idx) - 1):
            start = sep_idx[i] + 1
            end = sep_idx[i + 1]
            id_data = []
            # 开始转换，拿到每个
            for feature in samples.columns:
                id_data.append(item2id(list(samples[feature])[start:end], map_dict[feature][2]))
            result.append(id_data)
        # 去掉换行符
        if len(result[-1][0]) == 1:
            result = result[:-1]
        # 拼接长短句，数据增强
        two = []
        for i in range(len(result) - 1):
            first = result[i]
            second = result[i + 1]
            # 拼接两个
            two.append([first[k] + second[k] for k in range(len(first))])
        # 拼接三个
        three = []
        for i in range(len(result) - 2):
            first = result[i]
            second = result[i + 1]
            third = result[i + 2]
            # 拼接三个
            three.append([first[k] + second[k] + third[k] for k in range(len(first))])
        results.extend(result + two + three)
    # 保存到文件, 这里将训练集分为train和dev
    if name == 'train':
        split_ratio = [0.8, 0.2]
        total = len(results)
        p1 = int(total * split_ratio[0])
        p2 = int(total * (split_ratio[0] + split_ratio[1]))
        with open(f'datas/prepare_data/train.pkl', 'wb') as f:
            pickle.dump(results[:p1], f)
        with open(f'datas/prepare_data/dev.pkl', 'wb') as f:
            pickle.dump(results[p1:p2], f)
    else:
        with open(f'datas/prepare_data/test.pkl', 'wb') as f:
            pickle.dump(results, f)


# batch管理对象
class BatchManager(object):
    def __init__(self, batch_size, name='train'):
        # 这里就直接读取文件了
        with open(f'datas/prepare_data/' + name + '.pkl', 'rb') as f:
            data = pickle.load(f)
        # 初始化排序和填充
        self.batch_data = self.sort_pad(data, batch_size)
        # 计算总的batch长度
        self.len_data = len(self.batch_data)

    def sort_pad(self, data, batch_size):
        # 计算总共有多少个批次
        num_batch = int(math.ceil(len(data) / batch_size))
        # 安装句子长度排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        # 获取batch
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * int(batch_size): (i + 1) * int(batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        chars, bounds, flags, radicals, pinyins, targets = [], [], [], [], [], []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            char, bound, flag, target, radical, pinyin = line
            # 需要填充的个数
            padding = [0] * (max_length - len(char))
            chars.append(char + padding)
            bounds.append(bound + padding)
            flags.append(flag + padding)
            targets.append(target + padding)
            radicals.append(radical + padding)
            pinyins.append(pinyin + padding)
        return [chars, bounds, flags, radicals, pinyins, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


# 获取日志文件
def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# 创建对应的文件夹
def make_path(param):
    # 预测结果集文件夹
    if not os.path.isdir(param.result_path):
        os.makedirs(param.result_path)
    # 模型保存文件夹
    if not os.path.isdir(param.ckpt_path):
        os.makedirs(param.ckpt_path)
    # 日志文件
    if not os.path.isdir(param.log_dir):
        os.makedirs(param.log_dir)


# 测试结果写入文件
def test_ner(results, path):
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w", encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")
        f.writelines(to_write)
    # 返回评估报告
    eval_lines = return_report(output_file)
    return eval_lines


# 映射函数
def create_mapping(dico):
    """
    创造一个词典映射
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


# 使用预训练好的词典来扩充字典
def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    :param dictionary:频率字典
    :param ext_emb_path:预训练好的向量
    :param sentence:对应的词列表
    :return:
    """
    assert os.path.isfile(ext_emb_path)
    # 加载已经预训练好的词向量
    pretrained = set([line.rstrip().split()[0].strip()
                      for line in codecs.open(ext_emb_path, 'r', 'utf-8') if len(ext_emb_path) > 0])
    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    # 这里应该是在判断词在字典中没有，如果没有就分配为0
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [char, char.lower(), re.sub(r'\d', '0', char.lower())]) \
                    and char not in dictionary:
                dictionary[char] = 0
    # 重新生成词典映射
    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


# 读取预训练的词向量将其替换成新的
def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    加载预训练词向量
    :param emb_path:
    :param id_to_word:
    :param word_dim:
    :param old_weights:
    :return:
    """
    # 获取随机向量
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    # 无效词的统计
    emb_invalid = 0
    # 先遍历将值转化为浮点数
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            # 统计无效值
            emb_invalid += 1
    if emb_invalid > 0:
        # 打印无效向量
        print('WARNING: %i invalid lines' % emb_invalid)
    # 开始替换对应词典中存在的词对应的向量
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        # 从词典中取出词去找
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            # 寻找小写词，估计对应英文字母
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub(r'\d', '0', word.lower()) in pre_trained:
            # 寻找数字，对应数值且需要全部转化为0
            new_weights[i] = pre_trained[re.sub(r'\d', '0', word.lower())]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    # 打印统计信息
    print('%i / %i (%.4f%%) words have been initialized with pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words, 100. * (c_found + c_lower + c_zeros) / n_words))
    print('%i found directly, %i after lowercasing, %i after lowercasing + zero.' % (c_found, c_lower, c_zeros))
    # 返回新参数
    return new_weights


# 将结果写成JSON文件
def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entype = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append(
                {"word": char, "start": idx, "end": idx + 1, "type": entities_dict_chinese[tag[2:]]})
        elif tag[0] == "B":
            entype = entities_dict_chinese[tag[2:]]
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "O" or tag[0] == "S" or tag[0] == "B":
            if entity_name != "":
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx - 1,
                                         "type": entype})
                entity_name = ""
        idx += 1
    return item


if __name__ == '__main__':
    get_data_with_windows('train')
    get_data_with_windows('test')
    # lines = '我是中国人'
    # with open(f'datas/prepare_data/dict.pkl', 'rb') as f:
    #     map_dict = pickle.load(f)
    # # lines = input_from_line_with_feature(lines)
    # print(map_dict['bound'][2])
