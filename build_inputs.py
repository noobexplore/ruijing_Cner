#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/22 15:03
# @Author  : TheTAO
# @Site    : 
# @File    : build_inputs.py
# @Software: PyCharm
import pickle
import jieba.posseg as psg
from cnradical import Radical, RunOption


# 全角转半角
def full_to_half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


# 简单清理数据
def replace_html(stxt):
    stxt = stxt.replace('&quot;', '"')
    stxt = stxt.replace('&amp;', '&')
    stxt = stxt.replace('&lt;', '<')
    stxt = stxt.replace('&gt;', '>')
    stxt = stxt.replace('&nbsp;', ' ')
    stxt = stxt.replace("&ldquo;", "")
    stxt = stxt.replace("&rdquo;", "")
    stxt = stxt.replace("&mdash;", "")
    stxt = stxt.replace("\xa0", " ")
    return stxt


def input_from_line_with_feature(line):
    """
    此函数将单一输入句子进行实体识别，构造为具体如下形式
    [[[raw_text]], [[word]], [[bound]], [[flag]], [[label]], [[radical]], [[pinyin]]]
    这里多一列，到时候输入为[1:]
    :param line:输入的单一句子
    :param char_to_id:词典转索引
    :return:
    """
    with open(f'datas/prepare_data/dict.pkl', 'rb') as f:
        map_dict = pickle.load(f)

    def item2id(data, w2i):
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data]

    inputs = list()
    feature_names = ['word', 'bound', 'flag', 'radical', 'pinyin', 'label']
    line = full_to_half(line)
    line = replace_html(line)
    chars = [[char for char in line]]
    # 获取标签，先全部打上O
    tag_list = ['O' for _ in line]
    # 提取词性和词边界特征
    word_bounds = ['M' for _ in tag_list]  # 保存每个词的边界
    word_flags = []  # 保存词性
    # 遍历带词性的切分
    for word, flag in psg.cut(line):
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
    bounds = [word_bounds]
    flags = [word_flags]
    # 由于是测试将label置为空
    targets = [[]]
    # 获取偏旁和拼音特征
    radical = Radical(RunOption.Radical)
    pinyin = Radical(RunOption.Pinyin)
    # 这里循环迭代去获取，None的去填充
    radicals = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in line]]
    pinyins = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in line]]
    inputs.append(chars)
    inputs.append(bounds)
    inputs.append(flags)
    inputs.append(radicals)
    inputs.append(pinyins)
    inputs.append(targets)
    # 开始循环转化为数字索引
    id_inputs = [[line]]
    for i, feature in enumerate(feature_names):
        id_inputs.append([item2id(inputs[i][0], map_dict[feature][2])])
    return id_inputs[0][0], id_inputs[1:]


if __name__ == '__main__':
    lines = '我是中国人'
    id_input = input_from_line_with_feature(lines)
    print(id_input[0][0])
