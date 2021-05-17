#!/usr/bin/env python
# coding=utf-8
# Lint as: python3
"""Word-level language model corpus."""

import os
import pickle
import re

from nltk import DependencyGraph
from nltk.corpus import ptb

WORD_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
    'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
    'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
    'WP$', 'WRB', 'AUX', 'AUXG'
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
        head = int(head)
    except:
        # index can't be parsed as an integer, use default
        line_index, word, _, lemma, tag, _, head, rel, _, _ = cells
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

        total = sum(self.word2frq.values())
        cover = 0.0
        sorted_words = sorted(self.word2frq.items(), key=lambda item: item[1])[::-1]
        for k, v in sorted_words:
            if v >= thd and (k not in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1
                cover += v

        print('Number of words:', len(self.idx2word))
        print('Coverage:', cover / total)
        return len(self.idx2word)


class Corpus(object):
    """Word-level language model corpus."""

    def __init__(self, dataset='ptb', path='data/deps/', thd=5, test_only=False):
        """Initialization.

        Args:
          path: path to corpus location, the folder should include 'train.txt',
            'valid.txt' and 'test.txt'
          thd: tokens that appears less then thd times in train.txt will be replaced
            by <unk>
        """

        if not os.path.exists(path):
            os.mkdir(path)

        dict_file_name = os.path.join(path, dataset + '-dict.pkl')
        if os.path.exists(dict_file_name):
            print('Loading dictionary...')
            self.dictionary = pickle.load(open(dict_file_name, 'rb'))
            build_dict = False
        else:
            self.dictionary = Dictionary()
            build_dict = True

        label_file_name = os.path.join(path, dataset + '-label.pkl')
        if os.path.exists(label_file_name):
            print('Loading labels...')
            self.labels = pickle.load(open(label_file_name, 'rb'))
            build_label = False
        else:
            self.labels = Dictionary()
            build_label = True

        ood_test_file_ids = ['data/deps/en_gum-ud-test.conllu']

        if dataset == 'ptb':
            all_file_ids = ptb.fileids()
            train_file_ids = []
            valid_file_ids = []
            test_file_ids = []
            parser_test_file_ids = []
            for file_id in all_file_ids:
                if 'WSJ/00/WSJ_0000.MRG' <= file_id <= 'WSJ/20/WSJ_2099.MRG':
                    train_file_ids.append(os.path.expanduser(
                        "~") + '/nltk_data/corpora/ptb/' + file_id + '.dep')
                if 'WSJ/21/WSJ_2100.MRG' <= file_id <= 'WSJ/22/WSJ_2299.MRG':
                    valid_file_ids.append(os.path.expanduser(
                        "~") + '/nltk_data/corpora/ptb/' + file_id + '.dep')
                if 'WSJ/23/WSJ_2300.MRG' <= file_id <= 'WSJ/24/WSJ_2499.MRG':
                    test_file_ids.append(os.path.expanduser(
                        "~") + '/nltk_data/corpora/ptb/' + file_id + '.dep')
                if 'WSJ/23/WSJ_2300.MRG' <= file_id <= 'WSJ/23/WSJ_2399.MRG':
                    parser_test_file_ids.append(os.path.expanduser(
                        "~") + '/nltk_data/corpora/ptb/' + file_id + '.dep')
        else:
            path_1987 = '../data/LDC2000T43/1987/W7_%03d'
            path_1988 = '../data/LDC2000T43/1988/W8_%03d'
            path_1989 = '../data/LDC2000T43/1989/W9_%03d'

            train_path_XS = [path_1987 % id for id in [71, 122]] + \
                            [path_1988 % id for id in [54, 107]] + \
                            [path_1989 % id for id in [28, 37]]

            train_path_SM = [path_1987 % id for id in [35, 43, 48, 54, 61, 71, 77, 81, 96, 122]] + \
                            [path_1988 % id for id in [24, 54, 55, 59, 69, 73, 76, 79, 90, 107]] + \
                            [path_1989 % id for id in [12, 13, 15, 18, 21, 22, 28, 37, 38, 39]]

            train_path_MD = [path_1987 % id for id in [5, 10, 18, 21, 22, 26, 32, 35, 43, 47, 48, 49, 51, 54, 55, 56, 57, 61, 62, 65, 71, 77, 79, 81, 90, 96, 100, 105, 122, 125]] + \
                            [path_1988 % id for id in [12, 13, 14, 17, 23, 24, 33, 39, 40, 47, 48, 54, 55, 59, 69, 72, 73, 76, 78, 79, 83, 84, 88, 89, 90, 93, 94, 96, 102, 107]] + \
                            [path_1989 % id for id in range(12, 42)]

            train_path_LG = [path_1987 % id for id in range(3, 128)] + \
                            [path_1988 % id for id in range(3, 109)] + \
                            [path_1989 % id for id in range(12, 42)]

            if dataset == 'bllip-xs':
                train_path = train_path_XS
            elif dataset == 'bllip-sm':
                train_path = train_path_SM
            elif dataset == 'bllip-md':
                train_path = train_path_MD
            elif dataset == 'bllip-lg':
                train_path = train_path_LG
            valid_path = ['../data/LDC2000T43/1987/W7_001', '../data/LDC2000T43/1988/W8_001', '../data/LDC2000T43/1989/W9_010']
            test_path = ['../data/LDC2000T43/1987/W7_002', '../data/LDC2000T43/1988/W8_002', '../data/LDC2000T43/1989/W9_011']

            if not test_only:
                train_file_ids = []
                for p in train_path:
                    for file_name in os.listdir(p):
                        if file_name[-4:] == '.dep':
                            train_file_ids.append(os.path.join(p, file_name))
                            
            valid_file_ids = []
            for p in valid_path:
                for file_name in os.listdir(p)[:500]:
                    if file_name[-4:] == '.dep':
                        valid_file_ids.append(os.path.join(p, file_name))
            test_file_ids = []
            for p in test_path:
                for file_name in os.listdir(p)[:1000]:
                    if file_name[-4:] == '.dep':
                        test_file_ids.append(os.path.join(p, file_name))

            all_file_ids = ptb.fileids()
            parser_test_file_ids = []
            for file_id in all_file_ids:
                if 'WSJ/23/WSJ_2300.MRG' <= file_id <= 'WSJ/23/WSJ_2399.MRG':
                    parser_test_file_ids.append(os.path.expanduser(
                        "~") + '/nltk_data/corpora/ptb/' + file_id + '.dep')

        if not test_only:
            self.train, self.train_heads, self.train_labels \
                = self.tokenize(train_file_ids, build_dict, build_label, thd)
        self.valid, self.valid_heads, self.valid_labels \
            = self.tokenize(valid_file_ids)
        self.test, self.test_heads, self.test_labels \
            = self.tokenize(test_file_ids)
        self.parser_test, self.parser_test_heads, self.parser_test_labels \
            = self.tokenize(parser_test_file_ids)
        self.ood_test, self.ood_test_heads, self.ood_test_labels \
            = self.tokenize(ood_test_file_ids)

        if build_dict:
            print('Saving dictionary...')
            dict_file_name = os.path.join(path, dataset + '-dict.pkl')
            pickle.dump(self.dictionary, open(dict_file_name, 'wb'))

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in WORD_TAGS:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                words.append(w)
        return words

    def tokenize(self, file_ids, build_dict=False, build_label=False, thd=5):
        """Tokenizes a text file."""

        sens = []
        heads = []
        labels = []
        for file_id_i in file_ids:
            with open(file_id_i, 'r') as trg_file:
                trg_string = trg_file.read().strip()
                trg_string_list = trg_string.split('\n\n')
                for s in trg_string_list:
                    lines = s.split('\n')
                    new_lines = []
                    for l in lines:
                        if l[0] != '#':
                            new_lines.append(l)
                    s = '\n'.join(new_lines)

                    g = DependencyGraph(
                        s, top_relation_label='root', cell_extractor=extract_10_cells)
                    sen = []
                    sen_head = []
                    sen_label = []
                    address_mapping = []
                    for address in range(1, len(g.nodes)):
                        node = g.nodes[address]
                        if node['tag'] in WORD_TAGS:
                            w = node['word']
                            w = w.lower()
                            w = re.sub('[0-9]+', 'N', w)
                            sen.append(w)

                            head = node['head']
                            while (not g.nodes[head]['tag'] in WORD_TAGS) and (head > 0):
                                head = g.nodes[head]['head']
                            if head > 0:
                                sen_head.append(head - 1)
                            else:
                                sen_head.append(node['address'] - 1)
                            sen_label.append(node['rel'])
                            address_mapping.append(len(sen) - 1)
                        else:
                            address_mapping.append(-1)

                    sen_head = [address_mapping[ad] for ad in sen_head]

                    if len(sen) > 0:
                        sens.append(sen)
                        heads.append(sen_head)
                        labels.append(sen_label)

        if build_dict:
            # Add words to the dictionary
            for sen in sens:
                for word in sen:
                    self.dictionary.add_word(word)
            if thd > 1:
                self.dictionary.rebuild_by_freq(thd)

        if build_label:
            for sen_label in labels:
                for label in sen_label:
                    self.labels.add_word(label)

        # Tokenize file content
        ids_list = []
        labels_list = []
        for sen, sen_label in zip(sens, labels):
            ids = []
            label_ids = []
            for word, label in zip(sen, sen_label):
                ids.append(self.dictionary[word])
                label_ids.append(self.labels[label])
            ids_list.append(ids)
            labels_list.append(label_ids)

        return ids_list, heads, labels_list
