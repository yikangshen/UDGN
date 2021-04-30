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
import glob

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
            print("Building dictionary...")
            self.dictionary = Dictionary()
            build_dict = True

        label_file_name = os.path.join(path, 'label.pkl')
        if os.path.exists(label_file_name):
            print('Loading labels...')
            self.labels = pickle.load(open(label_file_name, 'rb'))
            build_label = False
        else:
            self.labels = Dictionary()
            build_label = True


        path_1987 = path + '/1987/W7_%03d'
        path_1988 = path + '/1988/W8_%03d'
        path_1989 = path + '/1989/W9_%03d'
        train_file_paths = [path_1987 % id for id in range(3, 128)] + \
                           [path_1988 % id for id in range(3, 109)] + \
                           [path_1989 % id for id in range(12, 42)]
        train_file_ids = []
        for file_path in train_file_paths:
            train_file_ids.extend(glob.glob(file_path + '/*.dep'))

        valid_file_ids = []
        for file_path in [path + '/1987/W7_001', \
                          path + '/1988/W8_001',
                          path + '/1989/W9_010']:
            valid_file_ids.extend(glob.glob(file_path + '/*.dep'))

        test_file_ids = []
        for file_path in [path + '/1987/W7_002',
                          path + '/1988/W8_002',
                          path + '/1989/W9_011']:
            test_file_ids.extend(glob.glob(file_path + '/*.dep'))

        parser_test_file_ids = test_file_ids
        print("train_file_ids", len(train_file_ids))
        print("valid_file_ids", len(valid_file_ids))
        print("test_file_ids", len(test_file_ids))
        print("parser_test_file_ids", len(parser_test_file_ids))

        self.train, self.train_heads, self.train_labels \
            = self.tokenize(train_file_ids, build_dict, build_label)
        self.valid, self.valid_heads, self.valid_labels \
            = self.tokenize(valid_file_ids)
        self.test, self.test_heads, self.test_labels \
            = self.tokenize(test_file_ids)
        self.parser_test, self.parser_test_heads, self.parser_test_labels \
            = self.tokenize(parser_test_file_ids)

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

    def tokenize(self, file_ids, build_dict=False, build_label=False, thd=5):
        """Tokenizes a text file."""
        import string
        sens = []
        heads = []
        labels = []
        for file_id_i in file_ids:
            with open(file_id_i, 'r') as trg_file:
                trg_string = trg_file.read().strip()
                trg_string_list = trg_string.split('\n\n')
                for s in trg_string_list:
                    g = DependencyGraph(
                        s, top_relation_label='root', cell_extractor=extract_10_cells)
                    sen = []
                    sen_head = []
                    sen_label = []
                    address_mapping = []

                    debug_print = []
                    fishy = False

                    for address in range(1, len(g.nodes)):
                        node = g.nodes[address]
                        debug_print.append(node['word'])
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
                            fishy = any(c in string.ascii_lowercase
                                        for c in node['word'].lower())
                            debug_print.append("[NW "+node['tag']+"]")
                            address_mapping.append(-1)
                        debug_print.append(' ')

                    sen_head = [address_mapping[ad] for ad in sen_head]

                    if len(sen) > 0:
                        if fishy:
                            raise Warning("\nOriginal sentence:" +
                                    ''.join(debug_print) + "\nProcessed: " +
                                    ''.join(sen) + "\n")
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


if __name__ == "__main__":
    corpus = Corpus("data/LDC2000T43", thd=5)
