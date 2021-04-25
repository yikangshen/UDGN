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
"""StructFormer and transformer model."""

from math import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import layers


def cumprod(x, reverse=False, exclusive=False):
    """cumulative product."""
    if reverse:
        x = x.flip([-1])

    if exclusive:
        x = F.pad(x[:, :, :-1], (1, 0), value=1)

    cx = x.cumprod(-1)

    if reverse:
        cx = cx.flip([-1])
    return cx


def cumsum(x, reverse=False, exclusive=False):
    """cumulative sum."""
    bsz, _, length = x.size()
    device = x.device
    if reverse:
        if exclusive:
            w = torch.ones([bsz, length, length], device=device).tril(-1)
        else:
            w = torch.ones([bsz, length, length], device=device).tril(0)
        cx = torch.bmm(x, w)
    else:
        if exclusive:
            w = torch.ones([bsz, length, length], device=device).triu(1)
        else:
            w = torch.ones([bsz, length, length], device=device).triu(0)
        cx = torch.bmm(x, w)
    return cx


def cummin(x, reverse=False, exclusive=False, max_value=1e9):
    """cumulative min."""
    if reverse:
        if exclusive:
            x = F.pad(x[:, :, 1:], (0, 1), value=max_value)
        x = x.flip([-1]).cummin(-1)[0].flip([-1])
    else:
        if exclusive:
            x = F.pad(x[:, :, :-1], (1, 0), value=max_value)
        x = x.cummin(-1)[0]
    return x


class StructFormer(nn.Module):
    """StructFormer model."""

    def __init__(self,
                 emb_size,
                 head_size,
                 nlayers,
                 ntokens,
                 nhead=8,
                 dropout=0.1,
                 dropatt=0,
                 pos_emb=False,
                 pad=0,
                 n_parser_layers=3,
                 weight_act='softmax',
                 relations='none',
                 share_params=False):
        """Initialization.

        Args:
          head_size: dimension of inputs and hidden states
          nlayers: number of layers
          ntokens: number of output categories
          nhead: number of self-attention heads
          dropout: dropout rate
          dropatt: drop attention rate
          pos_emb: bool, indicate whether use a learnable positional embedding
          pad: pad token index
          n_parser_layers: number of parsing layers
          conv_size: convolution kernel size for parser
          relations: relations that are used to compute self attention
          weight_act: relations distribution activation function
        """

        super(StructFormer, self).__init__()

        if relations == 'none':
            nrels = 0
        elif relations == 'type1':
            nrels = 2
        elif relations == 'type2':
            nrels = 2
        elif relations == 'type3':
            nrels = 4
        else:
            raise Exception

        self.drop = nn.Dropout(dropout)

        self.emb = nn.Embedding(ntokens, emb_size)
        self.parser_emb = nn.Embedding(ntokens, emb_size)
        if pos_emb:
            self.pos_emb = nn.Embedding(500, emb_size)

        if share_params:
            self.size_layers = 1
        else:
            self.size_layers = nlayers

        self.layers = nn.ModuleList([
            layers.TransformerLayer(
                emb_size, nhead, head_size,
                nrels=nrels, dropout=dropout, dropatt=dropatt)
            for _ in range(self.size_layers)])

        self.norm = nn.LayerNorm(emb_size)

        self.output_layer = nn.Linear(emb_size, ntokens)
        self.output_layer.weight = self.emb.weight

        self.parser_layers = nn.LSTM(emb_size, emb_size, n_parser_layers,
                                     dropout=dropout, batch_first=True, bidirectional=True)

        self.parser_ff = nn.Linear(emb_size * 2, emb_size * 2)

        self.n_parse_layers = n_parser_layers
        self.weight_act = weight_act
        self.nlayers = nlayers
        self.nhead = nhead
        self.ntokens = ntokens
        self.emb_size = emb_size
        self.pad = pad
        self.relations = relations

        self.init_weights()

    def init_weights(self):
        """Initialize token embedding and output bias."""
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.parser_emb.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'pos_emb'):
            self.pos_emb.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.fill_(0)

        init.xavier_uniform_(self.parser_ff.weight)
        init.zeros_(self.parser_ff.bias)

    def visibility(self, x):
        """Mask pad tokens."""
        visibility = (x != self.pad)
        visibility = visibility[:, None, :].expand(-1, x.size(1), -1)
        return visibility

    def parse(self, x, deps=None):
        """Parse input sentence.

        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          distance: syntactic distance
          height: syntactic height
        """

        mask = (x != self.pad)
        lengths = mask.sum(1).cpu().int()
        visibility = mask[:, None, :].expand(-1, x.size(1), -1)

        h = self.parser_emb(x)
        h = self.drop(h)
        h = pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.parser_layers(h)
        h, _ = pad_packed_sequence(h, batch_first=True)

        h = self.drop(h)
        parent, child = self.parser_ff(h).chunk(2, dim=-1)

        if deps is not None:
            bsz, length = x.size()
            p = torch.zeros((bsz, length, length), device=x.device)
            deps = deps.clamp(min=0)
            p.scatter_(2, deps[:, :, None], 1)
            return p, torch.zeros_like(p), h

        scaling = self.emb_size ** -0.5
        logits = torch.bmm(child, parent.transpose(1, 2)) * scaling
        logits = logits.masked_fill(~visibility, -inf)
        p = torch.softmax(logits, dim=-1)

        return p, logits

    def generate_mask(self, p):
        """Compute head and cibling distribution for each token."""

        bsz, length, _ = p.size()

        eye = torch.eye(length, device=p.device, dtype=torch.bool)
        eye = eye[None, :, :].expand((bsz, -1, -1))
        head = p.masked_fill(eye, 0)
        child = head.transpose(1, 2)

        # For better gradient
        att_mask = head + child - head * child
        
        ones = torch.ones_like(p)

        rels = []
        left = ones.tril(-1)
        right = ones.triu(1)
        if self.relations == 'none':
            rels = None
        elif self.relations == 'type1':
            rels = torch.stack([left, right], dim=-1)
        elif self.relations == 'type2':
            rels = torch.stack([head, child], dim=-1)
        elif self.relations == 'type3':
            rels0 = torch.stack([left, right], dim=-1)
            rels1 = torch.stack([left, right], dim=-1)
            rels = rels0[:, :, :, :, None] * rels1[:, :, :, None, :]
            rels = rels.view(bsz, length, length, -1)

        return att_mask, head, rels

    def encode(self, x, pos, att_mask, rels):
        """Structformer encoding process."""
        att_mask = att_mask[None, :, :, :, None].repeat(self.nlayers, 1, 1, 1, self.nhead)
        visibility = self.visibility(x)
        h = self.emb(x)
        if hasattr(self, 'pos_emb'):
            assert pos.max() < 500
            h = h + self.pos_emb(pos)
        h = self.drop(h)
        for i in range(self.nlayers):
            h = self.layers[i % self.size_layers](
                h, rels, attn_mask=att_mask[i % self.size_layers],
                key_padding_mask=visibility)
        return h

    def forward(self, x, pos, deps=None):
        """Pass the input through the encoder layer.

        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          output: probability distributions for missing tokens.
          state_dict: parsing results and raw output
        """

        batch_size, length = x.size()

        p, logp = self.parse(x, deps)
        att_mask, head, rels = self.generate_mask(p)

        raw_output = self.encode(x, pos, att_mask, rels)
        raw_output = self.norm(raw_output)
        raw_output = self.drop(raw_output)

        output = self.output_layer(raw_output)

        return output.view(batch_size * length, -1), \
            {'raw_output': raw_output, 'att_mask': att_mask,
             'head': head, 'root': raw_output[:, 0],
             'loghead': logp.view(batch_size * length, -1)}
