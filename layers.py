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
"""Convolution and transformer layers used by StructFormer."""

import math

import torch
from torch import nn
from torch.nn import init


def _get_activation_fn(activation):
    """Get specified activation function."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Conv1d(nn.Module):
    """1D convolution layer."""

    def __init__(self, hidden_size, kernel_size, dilation=1):
        """Initialization.

        Args:
          hidden_size: dimension of input embeddings
          kernel_size: convolution kernel size
          dilation: the spacing between the kernel points
        """
        super(Conv1d, self).__init__()

        if kernel_size % 2 == 0:
            padding = (kernel_size // 2) * dilation
            self.shift = True
        else:
            padding = ((kernel_size - 1) // 2) * dilation
            self.shift = False
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=padding,
            dilation=dilation)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x):
        """Compute convolution.

        Args:
          x: input embeddings
        Returns:
          conv_output: convolution results
        """

        if self.shift:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)[:, 1:]
        else:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MultiheadAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self,
                 embed_dim,
                 head_dim,
                 num_heads,
                 dropout=0.,
                 dropatt=0.,
                 bias=True):
        """Initialization.

        Args:
          embed_dim: dimension of input embeddings
          num_heads: number of self-attention heads
          dropout: dropout rate
          bias: bool, indicate whether include bias for linear transformations
          v_proj: bool, indicate whether project inputs to new values
          out_proj: bool, indicate whether project outputs to new values
          relative_bias: bool, indicate whether use a relative position based
            attention bias
        """

        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = head_dim * num_heads

        self.num_heads = num_heads
        self.drop = nn.Dropout(dropout)
        self.head_dim = head_dim
        self.value_dim = head_dim

        self.qk_proj = nn.Linear(embed_dim, self.hidden_dim * 2, bias=bias)
        self.vg_proj = nn.Linear(embed_dim, self.hidden_dim * 2, bias=bias)
        self.out_proj = nn.Linear(self.hidden_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize attention parameters."""

        init.xavier_uniform_(self.qk_proj.weight)
        if self.qk_proj.bias is not None:
            init.constant_(self.qk_proj.bias, 0.)

        init.xavier_uniform_(self.vg_proj.weight)
        if self.vg_proj.bias is not None:
            init.constant_(self.vg_proj.bias, 0.)

        init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, ctl=None, key_padding_mask=None, attn_mask=None):
        """Compute multi-head self-attention.

        Args:
          query: input embeddings
          key_padding_mask: 3D mask that prevents attention to certain positions
          attn_mask: 3D mask that rescale the attention weight at each position
        Returns:
          attn_output: self-attention output
        """

        bsz, length, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if ctl is None:
            q, k = self.qk_proj(query).chunk(2, dim=-1)
        else:
            q, k = self.qk_proj(ctl).chunk(2, dim=-1)
        v, g = self.vg_proj(query).chunk(2, dim=-1)

        q = q.contiguous().view(bsz, length, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = k.contiguous().view(bsz, length, self.num_heads,
                                self.head_dim).transpose(1, 2)
        v = v.contiguous().view(bsz, length, self.num_heads,
                                self.head_dim).transpose(1, 2)
        g = g.contiguous().view(bsz, length, self.num_heads,
                                self.head_dim).transpose(1, 2)

        attn_output_weights = torch.einsum('bhid,bhjd->bhij', q, k)
        assert list(attn_output_weights.size()) == [
            bsz, self.num_heads, length, length]

        if attn_mask is None:
            if key_padding_mask is not None:
                attn_output_weights.masked_fill_(
                    ~key_padding_mask[:, None, :, :], -math.inf)
            scaling = self.head_dim ** -0.5
            attn_output_weights = torch.softmax(
                attn_output_weights * scaling, dim=2)
        else:
            assert list(attn_mask.size()) == [
                bsz, self.num_heads, length, length]
            scaling = self.head_dim ** -0.5
            attn_output_weights = attn_output_weights * scaling
            attn_output_weights = torch.softmax(
                attn_output_weights, dim=1) * attn_mask
            if key_padding_mask is not None:
                attn_output_weights.masked_fill_(
                    ~key_padding_mask[:, None, :, :], 0)

        attn_output = torch.einsum(
            'bhij,bhjd->bhid', attn_output_weights, torch.tanh(v))
        gated_output = attn_output * torch.sigmoid(g)
        gated_output = gated_output.transpose(1, 2).contiguous().view(
            bsz, length, self.hidden_dim)

        output = self.out_proj(gated_output)

        return output


class TransformerLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network."""

    def __init__(self,
                 d_model,
                 nhead,
                 d_hidden=64,
                 dropout=0.1,
                 dropatt=0.1):
        """Initialization.

        Args:
          d_model: dimension of inputs
          nhead: number of self-attention heads
          dim_feedforward: dimension of hidden layer in feedforward layer
          dropout: dropout rate
          dropatt: drop attention rate
          activation: activation function
          relative_bias: bool, indicate whether use a relative position based
            attention bias
        """

        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model, d_hidden, nhead, dropout=dropout, dropatt=dropatt)

        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead

    def forward(self, src, ctl=None, attn_mask=None, key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
          src: the sequence to the encoder layer (required).
          attn_mask: the mask for the src sequence (optional).
          key_padding_mask: the mask for the src keys per batch (optional).
        Returns:
          src3: the output of transformer layer, share the same shape as src.
        """
        src2 = self.self_attn(
            src, ctl=ctl, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)

        src2 = src + self.dropout(src2)
        return src2
