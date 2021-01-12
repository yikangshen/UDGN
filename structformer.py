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


class Transformer(nn.Module):
    """Transformer model."""

    def __init__(self,
                 hidden_size,
                 nlayers,
                 ntokens,
                 nhead=8,
                 dropout=0.1,
                 dropatt=0.1,
                 relative_bias=True,
                 pos_emb=False,
                 pad=0):
        """Initialization.

        Args:
          hidden_size: dimension of inputs and hidden states
          nlayers: number of layers
          ntokens: number of output categories
          nhead: number of self-attention heads
          dropout: dropout rate
          dropatt: drop attention rate
          relative_bias: bool, indicate whether use a relative position based
            attention bias
          pos_emb: bool, indicate whether use a learnable positional embedding
          pad: pad token index
        """

        super(Transformer, self).__init__()

        self.drop = nn.Dropout(dropout)

        self.emb = nn.Embedding(ntokens, hidden_size)
        if pos_emb:
            self.pos_emb = nn.Embedding(500, hidden_size)

        self.layers = nn.ModuleList([
            layers.TransformerLayer(hidden_size, nhead, hidden_size * 4, dropout,
                                    dropatt=dropatt, relative_bias=relative_bias)
            for _ in range(nlayers)])

        self.norm = nn.LayerNorm(hidden_size)

        self.output_layer = nn.Linear(hidden_size, ntokens)
        self.output_layer.weight = self.emb.weight

        self.init_weights()

        self.nlayers = nlayers
        self.nhead = nhead
        self.ntokens = ntokens
        self.hidden_size = hidden_size
        self.pad = pad

    def init_weights(self):
        """Initialize token embedding and output bias."""
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'pos_emb'):
            self.pos_emb.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.fill_(0)

    def visibility(self, x, device):
        """Mask pad tokens."""
        visibility = (x != self.pad).float()
        visibility = visibility[:, None, :].expand(-1, x.size(1), -1)
        visibility = torch.repeat_interleave(visibility, self.nhead, dim=0)
        return visibility.log()

    def encode(self, x, pos):
        """Standard transformer encode process."""
        h = self.emb(x)
        if hasattr(self, 'pos_emb'):
            h = h + self.pos_emb(pos)
        h_list = []
        visibility = self.visibility(x, x.device)

        for i in range(self.nlayers):
            h_list.append(h)
            h = self.layers[i](
                h.transpose(0, 1), key_padding_mask=visibility).transpose(0, 1)

        output = h
        h_array = torch.stack(h_list, dim=2)

        return output, h_array

    def forward(self, x, pos):
        """Pass the input through the encoder layer.

        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          output: probability distributions for missing tokens.
          state_dict: parsing results and raw output
        """

        batch_size, length = x.size()

        raw_output, _ = self.encode(x, pos)
        raw_output = self.norm(raw_output)
        raw_output = self.drop(raw_output)

        output = self.output_layer(raw_output)
        return output.view(batch_size * length, -1), {'raw_output': raw_output, }


class StructFormer(Transformer):
    """StructFormer model."""

    def __init__(self,
                 hidden_size,
                 nlayers,
                 ntokens,
                 nhead=8,
                 dropout=0.1,
                 dropatt=0.1,
                 relative_bias=False,
                 pos_emb=False,
                 pad=0,
                 n_parser_layers=4,
                 conv_size=9,
                 relations=('head', 'child'),
                 weight_act='softmax'):
        """Initialization.

        Args:
          hidden_size: dimension of inputs and hidden states
          nlayers: number of layers
          ntokens: number of output categories
          nhead: number of self-attention heads
          dropout: dropout rate
          dropatt: drop attention rate
          relative_bias: bool, indicate whether use a relative position based
            attention bias
          pos_emb: bool, indicate whether use a learnable positional embedding
          pad: pad token index
          n_parser_layers: number of parsing layers
          conv_size: convolution kernel size for parser
          relations: relations that are used to compute self attention
          weight_act: relations distribution activation function
        """

        super(StructFormer, self).__init__(
            hidden_size,
            nlayers,
            ntokens,
            nhead=nhead,
            dropout=dropout,
            dropatt=dropatt,
            relative_bias=relative_bias,
            pos_emb=pos_emb,
            pad=pad)

        self.parser_layers = nn.ModuleList([
            nn.Sequential(layers.Conv1d(hidden_size, conv_size),
                          nn.LayerNorm(hidden_size, elementwise_affine=False),
                          nn.Tanh()) for i in range(n_parser_layers)])

        self.parent_ff = nn.Linear(hidden_size, hidden_size)
        self.child_ff = nn.Linear(hidden_size, hidden_size)

        n_rel = len(relations)
        self._rel_weight = nn.Parameter(torch.zeros((nlayers, nhead, n_rel)))
        self._rel_weight.data.normal_(0, 0.1)

        self._scaler = nn.Parameter(torch.zeros(2))

        self.n_parse_layers = n_parser_layers
        self.weight_act = weight_act
        self.relations = relations

    @property
    def scaler(self):
        return self._scaler.exp()

    @property
    def rel_weight(self):
        if self.weight_act == 'sigmoid':
            return torch.sigmoid(self._rel_weight)
        elif self.weight_act == 'softmax':
            return torch.softmax(self._rel_weight, dim=-1)

    def parse(self, x, pos):
        """Parse input sentence.

        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          distance: syntactic distance
          height: syntactic height
        """

        mask = (x != self.pad)
        visibility = mask[:, None, :].expand(-1, x.size(1), -1)

        h = self.emb(x)
        for i in range(self.n_parse_layers):
            h = h.masked_fill(~mask[:, :, None], 0)
            h = self.parser_layers[i](h)

        parent = self.parent_ff(h)
        child = self.child_ff(h)

        logits = torch.bmm(child, parent.transpose(1,2))
        scaling = self.hidden_size ** -0.5
        logits = (logits * scaling).masked_fill(~visibility, -inf)
        p = torch.softmax(logits * scaling, dim=-1)

        return p

    def generate_mask(self, head):
        """Compute head and cibling distribution for each token."""

        bsz, length, _ = head.size()

        eye = torch.eye(length, device=head.device, dtype=torch.bool)
        eye = eye[None, :, :].expand((bsz, -1, -1))

        head = head.masked_fill(eye, 0)
        child = head.transpose(1, 2)
        cibling = torch.bmm(head, child).masked_fill(eye, 0)

        rel_list = []
        if 'head' in self.relations:
            rel_list.append(head)
        if 'child' in self.relations:
            rel_list.append(child)
        if 'cibling' in self.relations:
            rel_list.append(cibling)

        rel = torch.stack(rel_list, dim=1)

        rel_weight = self.rel_weight

        dep = torch.einsum('lhr,brij->lbhij', rel_weight, rel)
        att_mask = dep.reshape(self.nlayers, bsz * self.nhead, length, length)

        return att_mask, cibling, head

    def encode(self, x, pos, att_mask):
        """Structformer encoding process."""

        visibility = self.visibility(x, x.device)
        h = self.emb(x)
        if hasattr(self, 'pos_emb'):
            assert pos.max() < 500
            h = h + self.pos_emb(pos)
        for i in range(self.nlayers):
            h = self.layers[i](
                h.transpose(0, 1), attn_mask=att_mask[i],
                key_padding_mask=visibility).transpose(0, 1)
        return h

    def forward(self, x, pos):
        """Pass the input through the encoder layer.

        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          output: probability distributions for missing tokens.
          state_dict: parsing results and raw output
        """

        batch_size, length = x.size()

        p = self.parse(x, pos)
        att_mask, cibling, head = self.generate_mask(p)

        raw_output = self.encode(x, pos, att_mask)
        raw_output = self.norm(raw_output)
        raw_output = self.drop(raw_output)

        output = self.output_layer(raw_output)

        return output.view(batch_size * length, -1), \
            {'raw_output': raw_output, 'cibling': cibling, 'head': head}
