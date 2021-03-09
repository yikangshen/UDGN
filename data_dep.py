# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Word-level language model corpus."""

import os
import pickle
import re

import nltk
from nltk import DependencyGraph
from nltk.corpus import ptb

WORD_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
    'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
    'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
    'WP$', 'WRB'
]
CURRENCY_TAGS_WORDS = ['#', '$', 'C$', 'A$']
ELLIPSIS = [
    '*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*'
]
PUNCTUATION_TAGS = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
PUNCTUATION_WORDS = [
    '.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!',
    '...', '-LCB-', '-RCB-'
]


def extract_10_cells(cells, index):
    line_index, word, lemma, tag, _, head, rel, _, _, _ = cells
    try:
        index = int(line_index)
    except ValueError:
        # index can't be parsed as an integer, use default
        pass
    return index, word, lemma, tag, tag, '', head, rel


class Dictionary(object):
    """Dictionary for language model."""

    def __init__(self):
        self.word2idx = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
        self.idx2word = ['<unk>', '<pad>', '<mask>']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if item in self.word2idx:
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def rebuild_by_freq(self, thd=3):
        """Prune low frequency words."""
        self.word2idx = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
        self.idx2word = ['<unk>', '<pad>', '<mask>']

        for k, v in self.word2frq.items():
            if v >= thd and (k not in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print('Number of words:', len(self.idx2word))
        return len(self.idx2word)


class Corpus(object):
    """Word-level language model corpus."""

    def __init__(self, path, thd=0):
        """Initialization.

        Args:
          path: path to corpus location, the folder should include 'train.txt',
            'valid.txt' and 'test.txt'
          thd: tokens that appears less then thd times in train.txt will be replaced
            by <unk>
        """

        if not os.path.exists(path):
            os.mkdir(path)

        dict_file_name = os.path.join(path, 'dict.pkl')
        if os.path.exists(dict_file_name):
            print('Loading dictionary...')
            self.dictionary = pickle.load(open(dict_file_name, 'rb'))
            build_dict = False
        else:
            self.dictionary = Dictionary()
            build_dict = True

        all_file_ids = ptb.fileids()
        train_file_ids = []
        valid_file_ids = []
        test_file_ids = []
        rest_file_ids = []
        for file_id in all_file_ids:
            if 'WSJ/00/WSJ_0200.MRG' <= file_id <= 'WSJ/21/WSJ_2199.MRG':
                train_file_ids.append(file_id)
            if 'WSJ/22/WSJ_2200.MRG' <= file_id <= 'WSJ/22/WSJ_2299.MRG':
                valid_file_ids.append(file_id)
            if 'WSJ/23/WSJ_2300.MRG' <= file_id <= 'WSJ/23/WSJ_2399.MRG':
                test_file_ids.append(file_id)
            elif ('WSJ/00/WSJ_0000.MRG' <= file_id <= 'WSJ/01/WSJ_0199.MRG') or \
                    ('WSJ/24/WSJ_2400.MRG' <= file_id <= 'WSJ/24/WSJ_2499.MRG'):
                rest_file_ids.append(file_id)

        self.train, self.train_heads \
            = self.tokenize(train_file_ids, build_dict=build_dict)
        self.valid, self.valid_heads \
            = self.tokenize(valid_file_ids)
        self.test, self.test_heads \
            = self.tokenize(test_file_ids)

        if build_dict:
            print('Saving dictionary...')
            dict_file_name = os.path.join(path, 'dict.pkl')
            pickle.dump(self.dictionary, open(dict_file_name, 'wb'))

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in WORD_TAGS:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                words.append(w)
        return words

    def tokenize(self, file_ids, build_dict=False, thd=5):
        """Tokenizes a text file."""

        sens = []
        heads = []
        labels = []
        for file_id_i in file_ids:
            file_id_i = '/home/hmwv1114/nltk_data/corpora/ptb/' + file_id_i + '.dep'
            with open(file_id_i, 'r') as trg_file:
                trg_string = trg_file.read().strip()
                trg_string_list = trg_string.split('\n\n')
                for s in trg_string_list:
                    g = DependencyGraph(s, top_relation_label='root', cell_extractor=extract_10_cells)
                    sen = []
                    head = []
                    label = []
                    for address in g.nodes:
                        if address > 0:
                            node = g.nodes[address]
                            w = node['word']
                            w = w.lower()
                            w = re.sub('[0-9]+', 'N', w)
                            sen.append(w)
                            if node['head'] > 0:
                                head.append(node['head'] - 1)
                            else:
                                head.append(node['address'] - 1)
                            label.append(node['rel'])

                    if len(sen) > 0:
                        sens.append(sen)
                        heads.append(head)
                        labels.append(label)

        if build_dict:
            # Add words to the dictionary
            for sen in sens:
                for word in sen:
                    self.dictionary.add_word(word)
            if thd > 1:
                self.dictionary.rebuild_by_freq(thd)

        # Tokenize file content
        ids_list = []
        for sen in sens:
            ids = []
            for word in sen:
                ids.append(self.dictionary[word])
            ids_list.append(ids)

        return ids_list, heads
