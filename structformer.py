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


class Transformer(nn.Module):
    """Transformer model."""

    def __init__(self,
                 emb_size,
                 nlayers,
                 ntokens,
                 nhead=8,
                 dropout=0.1,
                 dropatt=0,
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

        self.emb = nn.Embedding(ntokens, emb_size)
        if pos_emb:
            self.pos_emb = nn.Embedding(500, emb_size)

        self.layers = nn.ModuleList([
            layers.TransformerLayer(emb_size, nhead, emb_size * 4, dropout,
                                    dropatt=dropatt)
            for _ in range(nlayers)])

        self.norm = nn.LayerNorm(emb_size)

        self.output_layer = nn.Linear(emb_size, ntokens)
        self.output_layer.weight = self.emb.weight

        self.init_weights()

        self.nlayers = nlayers
        self.nhead = nhead
        self.ntokens = ntokens
        self.hidden_size = emb_size
        self.pad = pad

        self.criterion = nn.CrossEntropyLoss()

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

    def forward(self, x, y, pos, deps=None):
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

        target_mask = y != self.pad
        output = self.output_layer(raw_output[target_mask])
        loss = self.criterion(output, y[target_mask])
        return loss, {
            'raw_output': raw_output, 
            'head': torch.zeros((batch_size, length, length), device=x.device), 
            'loghead': torch.zeros((batch_size * length, length), device=x.device)}


class StructFormer(Transformer):
    """StructFormer model."""

    def __init__(self,
                 emb_size,
                 nlayers,
                 ntokens,
                 nhead=8,
                 dropout=0.1,
                 dropatt=0,
                 pos_emb=False,
                 pad=0,
                 n_parser_layers=3,
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
            emb_size,
            nlayers,
            ntokens,
            nhead=nhead,
            dropout=dropout,
            dropatt=dropatt,
            pos_emb=pos_emb,
            pad=pad)

        self.parser_layers = nn.ModuleList([
            nn.Sequential(layers.Conv1d(emb_size, conv_size),
                          nn.LayerNorm(emb_size, elementwise_affine=False),
                          nn.Tanh()) for i in range(n_parser_layers)])

        self.distance_ff = nn.Sequential(
            layers.Conv1d(emb_size, 2),
            nn.LayerNorm(emb_size, elementwise_affine=False), nn.Tanh(),
            nn.Linear(emb_size, 1))

        self.height_ff = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LayerNorm(emb_size, elementwise_affine=False), nn.Tanh(),
            nn.Linear(emb_size, 1))

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
        elif self.weight_act == 'ones':
            return torch.ones_like(self._rel_weight)

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
        mask_shifted = F.pad(mask[:, 1:], (0, 1), value=0)

        h = self.emb(x)
        h = self.drop(h)
        for i in range(self.n_parse_layers):
            h = h.masked_fill(~mask[:, :, None], 0)
            h = self.parser_layers[i](h)
            h = self.drop(h)

        height = self.height_ff(h).squeeze(-1)
        height.masked_fill_(~mask, -1e9)

        distance = self.distance_ff(h).squeeze(-1)
        distance.masked_fill_(~mask_shifted, 1e9)

        # Calbrating the distance and height to the same level
        length = distance.size(1)
        height_max = height[:, None, :].expand(-1, length, -1)
        height_max = torch.cummax(
            height_max.triu(0) - torch.ones_like(height_max).tril(-1) * 1e9,
            dim=-1)[0].triu(0)

        margin_left = torch.relu(
            F.pad(distance[:, :-1, None], (0, 0, 1, 0), value=1e9) - height_max)
        margin_right = torch.relu(distance[:, None, :] - height_max)
        margin = torch.where(margin_left > margin_right, margin_right,
                             margin_left).triu(0)

        margin_mask = torch.stack(
            [mask_shifted] + [mask] * (length - 1), dim=1)
        margin.masked_fill_(~margin_mask, 0)
        margin = margin.max()

        distance = distance - margin

        return distance, height

    def compute_block(self, distance, height):
        """Compute constituents from distance and height."""

        beta_logits = (distance[:, None, :] -
                       height[:, :, None]) * self.scaler[0]

        gamma = torch.sigmoid(-beta_logits)
        ones = torch.ones_like(gamma)

        block_mask_left = cummin(
            gamma.tril(-1) + ones.triu(0), reverse=True, max_value=1)
        block_mask_left = block_mask_left - F.pad(
            block_mask_left[:, :, :-1], (1, 0), value=0)
        block_mask_left.tril_(0)

        block_mask_right = cummin(
            gamma.triu(0) + ones.tril(-1), exclusive=True, max_value=1)
        block_mask_right = block_mask_right - F.pad(
            block_mask_right[:, :, 1:], (0, 1), value=0)
        block_mask_right.triu_(0)

        block_p = block_mask_left[:, :, :, None] * \
            block_mask_right[:, :, None, :]
        block = cumsum(block_mask_left).tril(0) + cumsum(
            block_mask_right, reverse=True).triu(1)

        return block_p, block

    def compute_head(self, height):
        """Estimate head for each constituent."""

        _, length = height.size()
        head_logits = height * self.scaler[1]
        index = torch.arange(length, device=height.device)

        mask = (index[:, None, None] <= index[None, None, :]) * (
            index[None, None, :] <= index[None, :, None])
        head_logits = head_logits[:, None, None,
                                  :].repeat(1, length, length, 1)
        head_logits.masked_fill_(~mask[None, :, :, :], -1e9)

        head_p = torch.softmax(head_logits, dim=-1)

        return head_p

    def generate_mask(self, x, distance, height):
        """Compute head and cibling distribution for each token."""

        bsz, length = x.size()

        eye = torch.eye(length, device=x.device, dtype=torch.bool)
        eye = eye[None, :, :].expand((bsz, -1, -1))

        block_p, block = self.compute_block(distance, height)
        head_p = self.compute_head(height)
        head_p = torch.einsum('blij,bijh->blh', block_p, head_p)
        head = head_p.masked_fill(eye, 0)
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

        return att_mask, (head_p + 1e-9).log(), head, block

    def encode(self, x, pos, att_mask):
        """Structformer encoding process."""

        visibility = self.visibility(x, x.device)
        h = self.emb(x)
        if hasattr(self, 'pos_emb'):
            assert pos.max() < 500
            h = h + self.pos_emb(pos)
        h = self.drop(h)
        for i in range(self.nlayers):
            h = self.layers[i](
                h.transpose(0, 1), attn_mask=att_mask[i],
                key_padding_mask=visibility).transpose(0, 1)
        return h

    def forward(self, x, y, pos, deps=None):
        """Pass the input through the encoder layer.
        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          output: probability distributions for missing tokens.
          state_dict: parsing results and raw output
        """

        batch_size, length = x.size()

        distance, height = self.parse(x, pos)
        att_mask, loghead, head, block = self.generate_mask(
            x, distance, height)

        raw_output = self.encode(x, pos, att_mask)
        raw_output = self.norm(raw_output)
        raw_output = self.drop(raw_output)

        target_mask = y != self.pad
        output = self.output_layer(raw_output[target_mask])
        loss = self.criterion(output, y[target_mask])

        return loss, \
            {'raw_output': raw_output, 'distance': distance, 'height': height,
             'head': head, 'block': block, 'loghead': loghead.view(batch_size * length, -1)}


class DSAN(nn.Module):
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
                 ntags=5,
                 relations='none',
                 detach_parser=False):
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

        super(DSAN, self).__init__()

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

        self.layers = nn.ModuleList([
            layers.DSANLayer(
                emb_size, nhead, head_size,
                nrels=nrels, dropout=dropout, dropatt=dropatt)
            for _ in range(nlayers)])

        self.norm = nn.LayerNorm(emb_size)

        self.output_layer = nn.Linear(emb_size, ntokens)
        self.output_layer.weight = self.emb.weight

        self.tag_emb = nn.Linear(ntags, emb_size, bias=False)

        self.tag = nn.Embedding(ntokens, ntags + 1)

        self.parser_layers = nn.LSTM(emb_size, emb_size, n_parser_layers,
                                     dropout=dropout, batch_first=True, bidirectional=True)

        self.parser_ff = nn.Linear(emb_size * 2, emb_size * 2)

        self.n_parse_layers = n_parser_layers
        self.nlayers = nlayers
        self.nhead = nhead
        self.ntokens = ntokens
        self.emb_size = emb_size
        self.pad = pad
        self.relations = relations
        self.detach_parser = detach_parser


        self.criterion = nn.CrossEntropyLoss()

        self.init_weights()

    def init_weights(self):
        """Initialize token embedding and output bias."""
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.parser_emb.weight.data.uniform_(-initrange, initrange)
        self.tag_emb.weight.data.uniform_(-initrange, initrange)
        self.tag.weight.data.zero_()
        if hasattr(self, 'pos_emb'):
            self.pos_emb.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.fill_(0)

        init.xavier_uniform_(self.parser_ff.weight)
        init.zeros_(self.parser_ff.bias)

    def parser_parameters(self):
        params = []
        params.extend(self.parser_emb.parameters())
        params.extend(self.parser_layers.parameters())
        params.extend(self.parser_ff.parameters())
        params.extend(self.tag.parameters())
        params.extend(self.tag_emb.parameters())
        return params

    def lm_parameters(self):
        params = []
        params.extend(self.output_layer.parameters())
        params.extend(self.layers.parameters())
        params.extend(self.norm.parameters())
        return params

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

        if deps is not None:
            bsz, length = x.size()
            p = torch.zeros((bsz, length, length), device=x.device)
            deps = deps.clamp(min=0)
            p.scatter_(2, deps[:, :, None], 1)
            return p, p.log()

        mask = (x != self.pad)
        lengths = mask.sum(1).cpu().int()
        visibility = mask[:, None, :].expand(-1, x.size(1), -1)

        # emb = self.parser_emb(x)

        tag = torch.softmax(self.tag(x), dim=-1)
        emb = self.tag_emb(tag[:, :, 1:]) + self.parser_emb(x) 

        h = self.drop(emb)
        h = pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.parser_layers(h)
        h, _ = pad_packed_sequence(h, batch_first=True)

        h = self.drop(h)
        parent, child = self.parser_ff(h).chunk(2, dim=-1)

        scaling = self.emb_size ** -0.5
        logits = torch.bmm(child, parent.transpose(1, 2)) * scaling
        logits = logits.masked_fill(~visibility, -inf)
        p = torch.softmax(logits, dim=-1)

        return p, tag, logits

    def generate_mask(self, p):
        """Compute head and cibling distribution for each token."""

        bsz, length, _ = p.size()

        eye = torch.eye(length, device=p.device, dtype=torch.bool)
        eye = eye[None, :, :].expand((bsz, -1, -1))
        head = p.masked_fill(eye, 0)
        child = head.transpose(1, 2)

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
            rels = F.normalize(rels, p=1, dim=-1)
        elif self.relations == 'type3':
            rels0 = torch.stack([left, right], dim=-1)
            rels1 = torch.stack([head, child], dim=-1)
            rels1 = F.normalize(rels1, p=1, dim=-1)
            rels = rels0[:, :, :, :, None] * rels1[:, :, :, None, :]
            rels = rels.view(bsz, length, length, -1)

        if self.detach_parser:
            att_mask = att_mask.detach()
            if rels is not None:
                rels = rels.detach()

        return att_mask, head, rels

    def encode(self, x, pos, att_mask, rels):
        """Structformer encoding process."""
        att_mask = (att_mask + 1e-6).log()
        visibility = self.visibility(x)
        h = self.emb(x)
        if hasattr(self, 'pos_emb'):
            assert pos.max() < 500
            h = h + self.pos_emb(pos)
        h = self.drop(h)
        all_layers = [h]
        for i in range(self.nlayers):
            h = self.layers[i](
                h, rels, attn_mask=att_mask,
                key_padding_mask=visibility)
            all_layers.append(h)
        return h, all_layers

    def forward(self, x, y, pos, deps=None):
        """Pass the input through the encoder layer.

        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          output: probability distributions for missing tokens.
          state_dict: parsing results and raw output
        """

        batch_size, length = x.size()

        p, tag, logp = self.parse(x, deps)
        att_mask, head, rels = self.generate_mask(p)

        raw_output, all_layers = self.encode(x, pos, att_mask, rels)
        raw_output = self.norm(raw_output)
        raw_output = self.drop(raw_output)

        target_mask = y != self.pad
        output = self.output_layer(raw_output[target_mask])
        loss = self.criterion(output, y[target_mask])
        return loss, \
            {'raw_output': raw_output, 'att_mask': att_mask,
             'head': head, 'tag': tag,
             'loghead': logp.view(batch_size * length, -1),
             'all_layers': all_layers}
