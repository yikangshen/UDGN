#!/usr/bin/env python
# coding=utf-8
# Lint as: python3
"""Test grammar induction performance of StructFormer."""

import argparse

import numpy
import torch

import data_dep
from ufal.chu_liu_edmonds import chu_liu_edmonds
from hinton import plot

import pickle
def mean(x):
    return sum(x) / len(x)

def compare_undirected(pred, deps):
    pred_pairs = set(tuple(sorted(x)) for x in enumerate(pred))
    deps_pairs = set(tuple(sorted(x)) for x in enumerate(deps))
    return len(pred_pairs & deps_pairs)

def dms(weights):
    weights = numpy.pad(weights, ((1, 0), (1, 0)), mode='constant')
    best_score = -numpy.inf
    best_heads = None
    for i in range(1, weights.shape[0]):
        W = weights.copy()
        W[:, 0] = numpy.nan
        W[i, 0] = 0.
        heads, tree_score = chu_liu_edmonds(W)
        if tree_score > best_score:
            best_score = tree_score
            best_heads = heads
    heads = numpy.array(best_heads) - 1
    pred = heads[1:]
    pred[pred == -1] = numpy.arange(pred.shape[0])[pred == -1]
    return pred


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
    label_count = len(corpus.labels.idx2word)
    dataset = zip(
        corpus.parser_test,
        corpus.parser_test_heads,
        corpus.parser_test_labels
    )

    correct = 0.0
    undir_correct = 0.0
    total = 0.0

    undirected = 0.0
    total_undirected = 0.0

    sum_x = 0.
    sum_y = 0.
    sum_xy = 0.

    for x, deps, rels in dataset:
        sen = [idx2word[idx] for idx in x]
        rels = torch.LongTensor(rels)
        data = torch.LongTensor([x]).to(device)
        pos = torch.LongTensor([list(range(len(x)))]).to(device)

        _, p_dict = parser(data, data, pos)
        mask = p_dict['att_mask']
        head = p_dict['head']
        attns = torch.cat(
            list(p[0] for p in p_dict['all_attn']),
            dim=-1
        )
        correct_dep_score = attns[pos[0], torch.tensor(deps)]
        rel_one_hot = torch.zeros((correct_dep_score.size(0), label_count),
                                  dtype=torch.float, device=attns.device)
        rel_one_hot[pos[0], rels] = 1.
        sum_x += correct_dep_score.sum(0)
        sum_y += rel_one_hot.sum(0)
        sum_xy += (correct_dep_score[:, :, None] * rel_one_hot[:, None, :]).sum(0)


        head = head.clone().squeeze(0).cpu().numpy()

        if mode == 'argmax':
            pred = numpy.argmax(head, axis=1)
        elif mode == 'tree':
            weights = numpy.log(head + 1e-9).astype(numpy.float)
            pred = dms(weights)
        else:
            raise Exception
        correct += (pred == deps).sum()
        undir_correct += compare_undirected(pred, deps)
        total += len(sen)

        thd = 0.2
        undirected += (mask[0, torch.range(0, len(sen)-1).long(), deps] > thd).sum()
        total_undirected += (mask > thd).sum() / 2

        nsens += 1

        if prt and nsens % 100 == 0:
            mask = mask.clone().squeeze(0).cpu().numpy()
            index = list(range(len(sen)))
            for id_i, word_i, pred_i, deps_i, mask_i, head_i in zip(
                    index, sen, pred, deps, mask, head):
                print('%2d\t%20s%3d%3d\t%s\t%s' %
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

    unsup_rel_cov = sum_xy / sum_y[None, :]
    pickle.dump((corpus.labels.idx2word, unsup_rel_cov),
                open('cov_mat.pkl', 'wb'))

    return float(dda), float(uda)


if __name__ == '__main__':
    marks = [' ', '-', '=']

    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

    argpr = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    argpr.add_argument(
        '--data',
        type=str,
        default='ptb',
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
    ptb_corpus = data_dep.Corpus(dataset=args.data)


    print('Evaluating...')
    if args.cuda:
        eval_device = torch.device('cuda:0')
    else:
        eval_device = torch.device('cpu')

    print('=' * 89)
    uas, uuas = test(model, ptb_corpus, eval_device, prt=args.print, mode=args.mode)
    print('Stanford Style: %.3f UAS, %.3f UUAS' % (uas, uuas))
    print('=' * 89)
