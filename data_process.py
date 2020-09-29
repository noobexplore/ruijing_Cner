#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/30 15:23
# @Author : TheTAO
# @Site : 
# @File : data_process.py.py
# @Software: PyCharm
import os
import re


def get_entities(dir):
    # 实体字典
    entities = {}
    files_list = os.listdir(dir)
    # 获取所有的文件名列表
    files = list(set([file.split('.')[0] for file in files_list]))
    # 遍历所有ann文件
    for file in files:
        # 构造每个文件的路径
        path = os.path.join(dir, file + '.ann')
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                name = line.split('\t')[1].split(' ')[0]
                if name in entities:
                    # 如果有就加一
                    entities[name] += 1
                else:
                    entities[name] = 1
    return entities


def get_labelencoder(entities):
    """
    返回标签和下标的映射
    :param entities:
    :return:
    """
    # 根据频率排序
    entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
    # 取出对应的实体
    entities = [x[0] for x in entities]
    # 构造标签字典
    id2label = ['O']
    for entity in entities:
        id2label.append('B-' + entity)
        id2label.append('I-' + entity)
    label2id = {id2label[i]: i for i in range(len(id2label))}
    return id2label, label2id


def ischinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False


def show_split_maxminlen(idxlist):
    # 先去重在排序
    idxlist = list(sorted(set([0, len(text)] + idxlist)))
    # 输出最长和最短，因为获取到的是首位
    lens = [idxlist[i + 1] - idxlist[i] for i in range(len(idxlist) - 1)]
    print(max(lens), min(lens))
    # 输出每一句话
    for i in range(len(idxlist) - 1):
        print(i, '|||||', text[idxlist[i]:idxlist[i + 1]])


def split_text(text):
    """
    此函数主要是为了将句子分开为以一句话为一个序列的网络输入
    首先是根据下面的一些标点去切分，但是要屏蔽掉一些情况
    :param text:
    :return:
    """
    # 记录分割的idx
    split_idx = []
    # 正则模式
    pattern1 = r'。|，|,|;|；|\.|\?'
    for m in re.finditer(pattern1, text):
        idx = m.span()[0]
        if text[idx - 1] == '\n':
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isdigit():  # 前后是数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isspace() and text[idx + 2].isdigit():  # 前数字 后空格 后后数字
            continue
        if text[idx - 1].islower() and text[idx + 1].islower():  # 前小写字母后小写字母
            continue
        if text[idx - 1].islower() and text[idx + 1].isdigit():  # 前小写字母后数字
            continue
        if text[idx - 1].isupper() and text[idx + 1].isdigit():  # 前大写字母后数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].islower():  # 前数字后小写字母
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isupper():  # 前数字后大写字母
            continue
        if text[idx + 1] in set('.。;；,，'):  # 前句号后句号
            continue
        if text[idx - 1].isspace() and text[idx - 2].isspace() and text[idx - 3] == 'C':  # HBA1C的问题
            continue
        if text[idx - 1].isspace() and text[idx - 2] == 'C':
            continue
        if text[idx - 1].isupper() and text[idx + 1].isupper():  # 前大些后大写
            continue
        if text[idx] == '.' and text[idx + 1:idx + 4] == 'com':  # 域名
            continue
        split_idx.append(idx + 1)
    # 这里找到一些特殊的词
    pattern2 = '\([一二三四五六七八九零十]\)|[一二三四五六七八九零十]、|'
    pattern2 += '注:|附录 |表 \d|Tab \d+|\[摘要\]|\[提要\]|表\d[^。，,;]+?\n|图 \d|Fig \d|'
    pattern2 += '\[Abstract\]|\[Summary\]|前  言|【摘要】|【关键词】|结    果|讨    论|'
    pattern2 += 'and |or |with |by |because of |as well as '
    # 遍历所有的模式
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if (text[idx:idx + 2] in ['or', 'by'] or text[idx:idx + 3] == 'and' or text[idx:idx + 4] == 'with') \
                and (text[idx - 1].islower() or text[idx - 1].isupper()):
            continue
        split_idx.append(idx)
    # 判断数字加.后面是否还有中文的这种情况
    pattern3 = '\n\d\.'
    for m in re.finditer(pattern3, text):
        idx = m.span()[0]
        # 判断是否为中文字符
        if ischinese(text[idx + 3]):
            split_idx.append(idx + 1)
    # 带括号数字的
    pattern4 = '\n\(\d\)'
    for m in re.finditer(pattern4, text):
        idx = m.span()[0]
        split_idx.append(idx + 1)
    # 对其索引排序
    split_idx = list(sorted(set([0, len(text)] + split_idx)))
    other_idx = []
    # 处理（一）xxx这种情况
    for i in range(len(split_idx) - 1):
        # 获取开始和结束符
        begin = split_idx[i]
        end = split_idx[i + 1]
        if text[begin] in '一二三四五六七八九零十' or \
                (text[begin] == '(' and text[begin + 1] in '一二三四五六七八九零十'):
            for j in range(begin, end):
                if text[j] == '\n':
                    other_idx.append(j + 1)
    # 处理完之后又加上other_idx
    split_idx += other_idx
    # 又需要进行新的排序
    split_idx = list(sorted(set([0, len(text)] + split_idx)))
    # 处理长句，长句子拆成短句子
    other_idx = []
    for i in range(len(split_idx) - 1):
        # 获取开始和结束
        b = split_idx[i]
        e = split_idx[i + 1]
        other_idx.append(b)
        # 如果长度超过150
        if e - b > 150:
            for j in range(b, e):
                # 保证句子长度在15以上
                if (j + 1 - other_idx[-1]) > 15:
                    # 如果为换行符
                    if text[j] == '\n':
                        other_idx.append(j + 1)
                    # 如果是空格后面跟数字的
                    if text[j] == ' ' and text[j - 1].isnumeric() and text[j + 1].isnumeric():
                        other_idx.append(j + 1)
    # 处理完之后又加上other_idx
    split_idx += other_idx
    # 又需要进行新的排序
    split_idx = list(sorted(set([0, len(text)] + split_idx)))
    # 干掉全是空格的句子
    for i in range(1, len(split_idx) - 1):
        idx = split_idx[i]
        # 处理空格，全部是空格的句子
        while idx > split_idx[i - 1] - 1 and text[idx - 1].isspace():
            idx -= 1
        split_idx[i] = idx
    # 又需要进行新的排序
    split_idx = list(sorted(set([0, len(text)] + split_idx)))
    # 因为需要跳过一些下标所以需要重新开辟
    temp_idx = []
    i = 0
    # 这里要跳跃一些下标，就可以合并了
    while i < len(split_idx) - 1:
        b = split_idx[i]
        e = split_idx[i + 1]
        # 先判断中英文字符
        num_ch = 0
        num_en = 0
        if e - b < 15:
            for ch in text[b:e]:
                if ischinese(ch):
                    num_ch += 1
                elif ch.islower() or ch.isupper():
                    num_en += 1
                # 如果长度够
                if num_ch + 0.5 * num_en > 5:
                    temp_idx.append(b)
                    i += 1
                    break
            if num_ch + 0.5 * num_en <= 5:
                # 合并后面的句子
                temp_idx.append(b)
                i += 2
        else:
            temp_idx.append(b)
            i += 1
    # 还需要重新排序
    split_idx = list(sorted(set([0, len(text)] + temp_idx)))
    # 返回最终切分结果
    result = []
    for i in range(len(split_idx) - 1):
        result.append(text[split_idx[i]:split_idx[i + 1]])
    # 检查切分是否正确
    s = ''
    for r in result:
        s += r
    # 最终的长度要等于text
    assert len(s) == len(text)
    return result


if __name__ == '__main__':
    datas_dir = './datas/ruijin_round1_train2_20181022/0.txt'
    with open(datas_dir, 'r', encoding='utf-8') as f:
        text = f.read()
        result = split_text(text)
        print(result)
