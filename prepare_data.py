#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/2 17:00
# @Author  : TheTao
# @Site    : 
# @File    : prepare_data.py
# @Software: PyCharm
import os
import shutil
import pandas as pd
import pickle
from collections import Counter
from data_process import split_text
import jieba.posseg as psg
from cnradical import Radical, RunOption
from random import shuffle
import multiprocessing as mp
from glob import glob

train_dir = 'ruijin_round1_train2_20181022'


def process_text(idx, split_method=None, split_name='train'):
    """
    读取文本，切割，然后打上标记并提取词边界、词性、偏旁部首、拼音等文本特征
    :param idx:文件的名字 不含扩展名
    :param split_method:切割文本的方法
    :param split_name:判断是保存训练集文件还是测试集
    :return:
    """
    data = {}
    # 获取句子
    if split_method is None:
        with open(f'./datas/{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        with open(f'./datas/{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = split_text(texts)
    data['word'] = texts
    # 获取标签，先全部打上O
    tag_list = ['O' for s in texts for _ in s]
    # 读取对应的ann文件
    tag = pd.read_csv(f'./datas/{train_dir}/{idx}.ann', header=None, sep='\t')
    for i in range(tag.shape[0]):
        # 获取实体类别以及起始位置
        tag_item = tag.iloc[i][1].split(' ')
        # 开始打标签
        cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])
        # 起始实体打上B
        tag_list[start] = 'B-' + cls
        # 其他的打上I
        for j in range(start + 1, end):
            tag_list[j] = 'I-' + cls
    # 做检查长度是否相等
    assert len([x for s in texts for x in s]) == len(tag_list)
    # 提取词性和词边界特征
    word_bounds = ['M' for item in tag_list]  # 保存每个词的边界
    word_flags = []  # 保存词性
    for text in texts:
        # 遍历带词性的切分
        for word, flag in psg.cut(text):
            # 单个词的时候
            if len(word) == 1:
                start = len(word_flags)
                word_bounds[start] = 'S'
                word_flags.append(flag)
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'
                word_flags += [flag] * len(word)
                # 这里end需要-1
                end = len(word_flags) - 1
                word_bounds[end] = 'E'
    # 这里保存词性，统一截断
    bounds = []
    flags = []
    tags = []
    start = 0
    end = 0
    for s in texts:
        ldx = len(s)
        end += ldx
        # 分句子显示
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        tags.append(tag_list[start:end])
        start += ldx
    data['bound'] = bounds
    data['flag'] = flags
    data['label'] = tags
    # 获取偏旁和拼音特征
    radical = Radical(RunOption.Radical)
    pinyin = Radical(RunOption.Pinyin)
    # 这里循环迭代去获取，None的去填充
    data['radical'] = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]
    data['pinyin'] = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]
    # 存储数据
    num_samples = len(texts)
    num_col = len(data.keys())

    dataset = []
    # 获取形如('中', 'B', 'ns', 'O', '丨', 'zhōng'), ('国', 'E', 'ns', 'O', '囗', 'guó')
    for i in range(num_samples):
        recoders = list(zip(*[list(v[i]) for v in data.values()]))  # *是解压的意思
        # 需要加入隔离符号对其隔离
        dataset += recoders + [['sep'] * num_col]
    # 最后一个不要
    dataset = dataset[:-1]
    # 转换成dataframe
    dataset = pd.DataFrame(dataset, columns=data.keys())
    # csv存储路径
    save_path = f'datas/prepare_data/{split_name}/{idx}.csv'

    # 现在开始可以处理换行符了
    def clean_word(w):
        if w == '\n':
            return 'LB'
        if w in [' ', '\t', '\u2003']:
            return 'SPACE'
        # 对所有的数字要统一处理
        if w.isdigit():
            return 'num'
        return w

    dataset['word'] = dataset['word'].apply(clean_word)
    dataset.to_csv(save_path, index=False, encoding='utf-8')


# 多进程处理
def multi_process(split_method=None, train_ratio=0.8):
    # 如果存在就先清空之后再创建
    if os.path.exists('./datas/prepare_data/'):
        # 删除对应的文件夹
        shutil.rmtree('./datas/prepare_data/')
    # 如果没有就创建
    if not os.path.exists('./datas/prepare_data/train/'):
        os.makedirs('./datas/prepare_data/train')
        os.makedirs('./datas/prepare_data/test')
    # 获取所有文件名
    idx = list(set([file.split('.')[0] for file in os.listdir('./datas/' + train_dir)]))
    # 打乱文件顺序
    shuffle(idx)
    # 拿到训练集的截止下标
    index = int(len(idx) * train_ratio)
    # 训练集
    train_idx = idx[:index]
    # 测试集
    test_idx = idx[index:]
    # 多进程操作
    num_cpus = mp.cpu_count()  # 获取cpu个数
    # 进程池
    pool = mp.Pool(num_cpus)
    results = []
    # 处理训练集
    for idx in train_idx:
        result = pool.apply_async(process_text, args=(idx, split_method, 'train'))
        results.append(result)
    for idx in test_idx:
        result = pool.apply_async(process_text, args=(idx, split_method, 'test'))
        results.append(result)
    pool.close()
    pool.join()


def mapping(data, threshold=10, is_word=False, sep='sep', is_label=False):
    count = Counter(data)
    if sep is not None:
        count.pop(sep)
    # 如果是词的话
    if is_word:
        count['PAD'] = 10000001
        count['UNK'] = 10000000
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        # 去掉频率小于threshold的值
        data = [x[0] for x in data if x[1] >= threshold]
        # 转化为映射
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        # 转化为映射
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    # 如果不是词的话
    else:
        # 句子的长度可能不一致，pad就是用来做填充用的
        count['PAD'] = 10000001
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        # 转化为映射
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    return id2item, item2id


def get_dict():
    # 存储字典
    map_dict = {}
    all_w, all_bound, all_flag, all_label, all_radical, all_pinyin = [], [], [], [], [], []
    # 遍历所有的文件
    for file in glob('./datas/prepare_data/train/*.csv') + glob('./datas/prepare_data/test/*.csv'):
        # 拿到对应的csv文件
        df = pd.read_csv(file, sep=',')
        # 分别对应获取到类别
        all_w += df['word'].tolist()
        all_bound += df['bound'].tolist()
        all_flag += df['flag'].tolist()
        all_label += df['label'].tolist()
        all_radical += df['radical'].tolist()
        all_pinyin += df['pinyin'].tolist()
    # 先映射词
    map_dict['word'] = mapping(all_w, threshold=10, is_word=True)
    map_dict['bound'] = mapping(all_bound)
    map_dict['flag'] = mapping(all_flag)
    map_dict['label'] = mapping(all_label, is_label=True)
    map_dict['radical'] = mapping(all_radical)
    map_dict['pinyin'] = mapping(all_pinyin)
    with open(f'datas/prepare_data/dict.pkl', 'wb') as f:
        pickle.dump(map_dict, f)


if __name__ == '__main__':
    multi_process(split_text)
    get_dict()
