# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging
from pprint import pprint

from scipy.stats import spearmanr, pearsonr



class STSEval(object):
    def loadFile(self, fpath):
        self.data = []
        for dataset in self.datasets:
            lines = io.open(fpath + '/STS.input.%s.txt' % dataset,
                            encoding='utf8').read().splitlines()
            scores = io.open(fpath + '/STS.gs.%s.txt' % dataset,
                             encoding='utf8').read().splitlines()
            for l, raw_score in zip(lines, scores):
                if raw_score != '':
                    data_item = {}
                    data_item['sentence1'], data_item['sentence2'] = l.split("\t")[:2]
                    data_item['label'] = float(raw_score)
                    self.data.append(data_item)


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)

class STSBenchmarkEval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.train = self.loadFile(os.path.join(taskpath, 'sts-train.csv'))
        self.dev = self.loadFile(os.path.join(taskpath, 'sts-dev.csv'))
        self.test = self.loadFile(os.path.join(taskpath, 'sts-test.csv'))
        self.data = self.test

    def loadFile(self, fpath):
        data = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for l in f:
                data_item = {}
                text = l.strip().split('\t')
                data_item['sentence1'] = text[5]
                data_item['sentence2'] = text[6]
                data_item['label'] = float(text[4])
                data.append(data_item)
        return data

class SICKRelatednessEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Relatedness*****\n\n')
        self.seed = seed
        self.train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        self.dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        self.test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.data = self.test

    def loadFile(self, fpath):
        skipFirstLine = True
        data = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    data_item = {}
                    data_item['sentence1'] = text[1]
                    data_item['sentence2'] = text[2]
                    data_item['label'] = float(text[3])
                    data.append(data_item)
        return data


if __name__ == "__main__":
    years = [12, 13, 14, 15, 16]
    taskpath ="/datadrive/shawn/code/SentEval/data/downstream/STS/STS%d-en-test"
    for year in years:
        sts = eval("STS%dEval" % year)(taskpath % year)
        for d in sts.data:
            pass
        print(len(sts.data))

    taskpath ="/datadrive/shawn/code/SentEval/data/downstream/STS/STSBenchmark"
    sts = STSBenchmarkEval(taskpath)
    for d in sts.data:
        pass
    print(len(sts.data))

    taskpath ="/datadrive/shawn/code/SentEval/data/downstream/SICK"
    sts = SICKRelatednessEval(taskpath)
    for d in sts.data:
        print(d)
    print(len(sts.data))

