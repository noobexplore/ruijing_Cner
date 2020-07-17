#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 9:19
# @Author  : TheTao
# @Site    : 
# @File    : main.py
# @Software: PyCharm
from test import test
from train import train
from params_utils import get_params


if __name__ == '__main__':
    params = get_params()
    is_train = params.train
    if is_train:
        train(params)
    else:
        test(params)
