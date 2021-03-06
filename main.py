#!/usr/bin/env python
# coding=utf-8
# Lint as: python3
"""Masked language model training script for UDGN."""

import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import data_dep
import models
from utils import batchify
from test_phrase_grammar import test

parser = argparse.ArgumentParser(
    description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument(
    '--data',
    type=str,
    default='ptb',
    help='location of the data corpus')
parser.add_argument(
    '--model',
    type=str,
    default='UDGN')
parser.add_argument('--dict_thd', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument(
    '--nemb', type=int, default=512, help='word embedding size')
parser.add_argument(
    '--head_size', type=int, default=128, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=8, help='number of layers')
parser.add_argument(
    '--n_parser_layers', type=int, default=3, help='number of layers')
parser.add_argument('--nheads', type=int, default=8, help='number of layers')
parser.add_argument(
    '--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument(
    '--parser_loss', action='store_true', help='Parser loss weight')
parser.add_argument('--ground_truth', action='store_true', help='use CUDA')
parser.add_argument(
    '--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument(
    '--batch_size', type=int, default=4096, metavar='N', help='batch size')
parser.add_argument(
    '--dropout',
    type=float,
    default=0.2,
    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument(
    '--parser_dropout',
    type=float,
    default=0.2,
    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument(
    '--dropatt',
    type=float,
    default=0.1,
    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument(
    '--mask_rate',
    type=float,
    default=0.3,
    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--pos_emb', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--nonmono', type=int, default=5, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--eval_runs', type=int, default=5)
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    metavar='N',
    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, '
              'so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed(args.seed)

if args.cuda:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

mask_bernoulli = torch.distributions.Bernoulli(args.mask_rate)


###############################################################################
# Load data
###############################################################################


def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer, scheduler], f)


def model_load(fn):
    global model, criterion, optimizer, scheduler
    with open(fn, 'rb') as f:
        model, criterion, optimizer, scheduler = torch.load(f)


print('Loading dataset...')
corpus = data_dep.Corpus(dataset=args.data, thd=args.dict_thd, test_only=args.test_only)

pad_token = corpus.dictionary.word2idx['<pad>']
mask_token = corpus.dictionary.word2idx['<mask>']
unk_token = corpus.dictionary.word2idx['<unk>']

val_data, val_heads = batchify(
    corpus.valid, corpus.valid_heads, args.batch_size, device, pad=pad_token)
test_data, test_heads = batchify(
    corpus.test, corpus.test_heads, args.batch_size, device, pad=pad_token)
ood_test_data, ood_heads = batchify(
    corpus.ood_test, corpus.ood_test_heads, args.batch_size, device, pad=pad_token)
parser_test_data, parser_test_heads = batchify(
    corpus.parser_test, corpus.parser_test_heads, args.batch_size, device, pad=pad_token)

###############################################################################
# Build the model
###############################################################################

criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
head_criterion = nn.CrossEntropyLoss(ignore_index=-1)

ntokens = len(corpus.dictionary)
print('Number of tokens: ', ntokens)

if args.model == 'structformer':
    model = models.StructFormer(
        emb_size=args.nemb,
        nlayers=args.nlayers,
        ntokens=ntokens,
        nhead=args.nheads,
        dropout=args.dropout,
        dropatt=args.dropatt,
        pos_emb=args.pos_emb,
        pad=pad_token,
        n_parser_layers=args.n_parser_layers)
elif args.model == 'transformer':
    model = models.Transformer(
        emb_size=args.nemb,
        nlayers=args.nlayers,
        ntokens=ntokens,
        nhead=args.nheads,
        dropout=args.dropout,
        dropatt=args.dropatt,
        pos_emb=args.pos_emb,
        pad=pad_token)
elif args.model == 'UDGN':
    model = models.UDGN(
        emb_size=args.nemb,
        head_size=args.head_size,
        nlayers=args.nlayers,
        ntokens=ntokens,
        nhead=args.nheads,
        dropout=args.dropout,
        parser_dropout=args.parser_dropout,
        dropatt=args.dropatt,
        pos_emb=args.pos_emb,
        pad=pad_token,
        n_parser_layers=args.n_parser_layers,
        detach_parser=args.parser_loss)

###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
###
if args.cuda:
    model = model.cuda()
###
params = list(model.parameters())
total_params = sum(np.prod(x.size()) for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Training code
###############################################################################


def mask_data(data):
    """randomly mask input sequence."""
    mask = mask_bernoulli.sample(data.shape).to(device).bool()
    mask = mask * (data != pad_token) * (data != unk_token)
    targets = data.masked_fill(~mask, pad_token)
    data = data.masked_fill(mask, mask_token)
    return data, targets


@torch.no_grad()
def evaluate_single_run(data_source, heads_source):
    """Evaluate the model on given dataset."""
    model.eval()
    total_loss = 0
    total_count = 0
    total_corr = 0
    total_words = 0
    for data, heads in zip(data_source, heads_source):
        data, targets = mask_data(data)
        pos = torch.arange(data.size(1), device=device)[None, :]

        loss, p_dict = model(data, targets, pos, heads if args.ground_truth else None)

        # loss = criterion(output, targets.reshape(-1))
        count = (targets != pad_token).float().sum().data
        total_loss += loss.data * count
        total_count += count

        head = p_dict['head']
        pred = head.argmax(-1)
        total_corr += (pred == heads).float().sum().data
        total_words += (heads > -1).float().sum().data

    return total_loss / total_count, total_corr / total_words


def evaluate(data_source, heads_source, nruns=5):
    test_loss_list = []
    test_masked_acc_list = []

    for _ in range(nruns):
        test_loss, test_masked_acc = evaluate_single_run(data_source, heads_source)
        test_loss_list.append(test_loss)
        test_masked_acc_list.append(test_masked_acc)

    test_loss = sum(test_loss_list) / nruns
    test_masked_acc = sum(test_masked_acc_list) / nruns

    return test_loss, test_masked_acc


@torch.no_grad()
def evaluate_parser(data_source, heads_source):
    """Evaluate the model on given dataset."""
    model.eval()
    total_corr = 0
    total_words = 0
    for data, heads in zip(data_source, heads_source):
        pos = torch.arange(data.size(1), device=device)[None, :]

        _, p_dict = model(data, data, pos, heads if args.ground_truth else None)

        head = p_dict['head']
        pred = head.argmax(-1)
        total_corr += (pred == heads).float().sum().data
        total_words += (heads > -1).float().sum().data

    return float(total_corr / total_words)


def train():
    """One epoch of training."""
    model.train()
    # Turn on training mode which enables dropout.
    total_loss = 0
    total_parser_loss = 0
    start_time = time.time()
    batch = 0
    train_data, train_heads = batchify(
        corpus.train, corpus.train_heads, args.batch_size, device, pad=pad_token, shuffle=True)
    while batch < len(train_data):
        data = train_data[batch]
        heads = train_heads[batch]
        data, targets = mask_data(data)
        pos = torch.arange(data.size(1), device=device)[None, :]

        optimizer.zero_grad()

        loss, p_dict = model(data, targets, pos, heads if args.ground_truth else None)
        # loss = criterion(output, targets.reshape(-1))
        logp_head = p_dict['loghead']
        parser_loss = head_criterion(logp_head, heads.reshape(-1))
        (loss + float(args.parser_loss) * parser_loss).backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += loss.data
        total_parser_loss += parser_loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            cur_parser_loss = total_parser_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | parser {:5.2f}'.format(
                      epoch, batch, len(
                          train_data), optimizer.param_groups[0]['lr'],
                      elapsed * 1000 / args.log_interval, cur_loss,
                      math.exp(cur_loss), cur_parser_loss))
            total_loss = 0
            total_parser_loss = 0
            start_time = time.time()
        ###
        batch += 1


# Loop over epochs.
lr = args.lr
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
if not args.test_only:
    try:
        if not args.resume:
            optimizer = torch.optim.Adam(
                params, lr=args.lr, eps=1e-9, weight_decay=args.wdecay)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', 0.5, patience=2, threshold=0)

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            train()

            val_loss, val_masked_acc = evaluate(val_data, val_heads, args.eval_runs)
            val_acc = evaluate_parser(val_data, val_heads)
            print('-' * 100)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | masked UAS {:5.3f} | valid UAS {:5.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss,
                    math.exp(val_loss), val_masked_acc, val_acc))
            print('-' * 100)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss
            scheduler.step(val_loss)

            print('PROGRESS: {}%'.format((epoch / args.epochs) * 100))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss, test_masked_acc = evaluate(test_data, test_heads, nruns=args.eval_runs)
ood_test_loss, _ = evaluate(ood_test_data, ood_test_data, nruns=args.eval_runs)

argmax_uas, argmax_uuas = test(model, corpus, torch.device('cuda:0') if args.cuda else torch.device('cpu'), mode='argmax')
tree_uas, tree_uuas = test(model, corpus, torch.device('cuda:0') if args.cuda else torch.device('cpu'), mode='tree')
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | masked UAS {:5.3f} '
      '| test UAS {:8.3f} / {:4.3f} | test UUAS {:8.3f} / {:4.3f} | ood test ppl {:8.2f}'.format(
          test_loss, math.exp(test_loss), test_masked_acc, 
          argmax_uas, tree_uas, argmax_uuas, tree_uuas, math.exp(ood_test_loss)))
print('=' * 89)
