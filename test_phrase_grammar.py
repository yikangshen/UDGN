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
"""Test grammar induction performance of StructFormer."""

import argparse

import numpy
import torch

import data_dep_blipp as data_dep
# import data_dep
import edmonds
from hinton import plot


def mean(x):
    return sum(x) / len(x)

def compare_undirected(pred, deps):
    pred_pairs = set(tuple(sorted(x)) for x in enumerate(pred))
    deps_pairs = set(tuple(sorted(x)) for x in enumerate(deps))
    return len(pred_pairs & deps_pairs)

@torch.no_grad()
def test(parser, corpus, device, prt=False, mode='tree'):
    """Compute UF1 and UAS scores.

    Args:
      parser: pretrained model
      corpus: labeled corpus
      device: cpu or gpu
      prt: bool, whether print examples
      gap: distance gap for building non-binary tree
    Returns:
      UF1: unlabeled F1 score for constituency parsing
    """
    parser.eval()

    dtree_list = []
    nsens = 0

    idx2word = corpus.dictionary.idx2word
    dataset = zip(corpus.parser_test, corpus.parser_test_heads)

    correct = 0.0
    undir_correct = 0.0
    total = 0.0

    undirected = 0.0
    total_undirected = 0.0

    for x, deps in dataset:
        sen = [idx2word[idx] for idx in x]
        data = torch.LongTensor([x]).to(device)
        pos = torch.LongTensor([list(range(len(x)))]).to(device)

        _, p_dict = parser(data, data, pos)
        mask = p_dict['att_mask']
        head = p_dict['head']

        head = head.clone().squeeze(0).cpu().numpy()

        if mode == 'argmax':
            pred = numpy.argmax(head, axis=1)
        elif mode == 'tree':
            pred = edmonds.single_root_msa(numpy.log(head))
        else:
            raise Exception

        correct += (pred == deps).sum()
        undir_correct += compare_undirected(pred, deps)
        total += len(sen)

        thd = 0.2
        undirected += (mask[0, torch.range(0, len(sen)-1).long(), torch.Tensor(deps).long()] > thd).sum()
        total_undirected += (mask > thd).sum() / 2

        nsens += 1

        if prt and nsens % 100 == 0:
            mask = mask.clone().squeeze(0).cpu().numpy()
            index = list(range(len(sen)))
            for id_i, word_i, pred_i, deps_i, mask_i, head_i in zip(
                    index, sen, pred, deps, mask, head):
                print('%2d\t%20s\t%2d\t%2d\t%s\t%s' %
                      (id_i, word_i, pred_i, deps_i,
                       plot(head_i, max_val=1), plot(mask_i, max_val=1.)))
            print()

    print('-' * 89)

    dda = correct / total
    uda = undir_correct / total

    prec = undirected / total_undirected
    reca = undirected / total
    f1 = 2 / (1 / prec + 1 / reca)

    print('Prec: %.3f, Reca: %.3f, F1: %.3f' % (prec, reca, f1))

    return float(dda), float(uda)


if __name__ == '__main__':
    marks = [' ', '-', '=']

    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

    argpr = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    argpr.add_argument(
        '--data',
        type=str,
        default='data/blipp_lg/',
        help='location of the data corpus')
    argpr.add_argument(
        '--checkpoint',
        type=str,
        default='PTB.pt',
        help='model checkpoint to use')
    argpr.add_argument(
        '--mode',
        type=str,
        default='tree',
        help='rule to find the head')
    argpr.add_argument('--seed', type=int, default=1111, help='random seed')
    argpr.add_argument('--print', action='store_true', help='use CUDA')
    argpr.add_argument('--cuda', action='store_true', help='use CUDA')
    args = argpr.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Load model
    print('Loading model...')
    with open(args.checkpoint, 'rb') as f:
        model, _, _, _ = torch.load(f, map_location='cpu')
        torch.cuda.manual_seed(args.seed)
        if args.cuda:
            model.cuda()

    # Load data
    print('Loading PTB dataset...')
    ptb_corpus = data_dep.Corpus(args.data)

    print('Evaluating...')
    if args.cuda:
        eval_device = torch.device('cuda:0')
    else:
        eval_device = torch.device('cpu')

    print('=' * 89)
    uas, uuas = test(model, ptb_corpus, eval_device, prt=args.print, mode=args.mode)
    print('Stanford Style: %.3f UAS, %.3f UUAS' % (uas, uuas))
    print('=' * 89)
