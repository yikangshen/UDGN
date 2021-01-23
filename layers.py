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
        init.kaiming_uniform_(self.conv.weight, nonlinearity='linear')
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
                 hidden_dim,
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
        self.hidden_dim = hidden_dim

        self.num_heads = num_heads
        self.drop = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.value_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, ("embed_dim must be "
                                                             "divisible by "
                                                             "num_heads")

        self.proj = nn.Linear(embed_dim, embed_dim * 2 + hidden_dim * 2, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize attention parameters."""

        init.kaiming_uniform_(self.proj.weight, nonlinearity='linear')
        init.constant_(self.proj.bias, 0.)

        init.kaiming_uniform_(self.out_proj.weight, nonlinearity='linear')
        init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key_padding_mask=None, attn_mask=None):
        """Compute multi-head self-attention.

        Args:
          query: input embeddings
          key_padding_mask: 3D mask that prevents attention to certain positions
          attn_mask: 3D mask that rescale the attention weight at each position
        Returns:
          attn_output: self-attention output
        """

        length, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v, g = self.proj(query).split(
            [self.embed_dim, self.embed_dim, self.hidden_dim, self.hidden_dim], dim=-1)

        q = q.contiguous().view(length, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)
        k = k.contiguous().view(length, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)
        v = v.contiguous().view(length, bsz * self.num_heads,
                                self.value_dim).transpose(0, 1)
        g = g.contiguous().view(length, bsz * self.num_heads,
                                self.value_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, length, length]

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights + key_padding_mask

        assert list(attn_mask.size()) == [bsz * self.num_heads, length, length]
        attn_output_weights = torch.sigmoid(attn_output_weights) * attn_mask

        attn_output = torch.bmm(attn_output_weights, v)
        gated_output = torch.tanh(attn_output) * torch.sigmoid(g)
        gated_output = gated_output.transpose(0, 1).contiguous().view(
            length, bsz, self.hidden_dim)

        output = self.out_proj(gated_output)

        return output


class TransformerLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network."""

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 dropatt=0.1,
                 activation="leakyrelu",
                 relative_bias=True):
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
            d_model, d_model, nhead, dropout=dropout, dropatt=dropatt)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
          src: the sequence to the encoder layer (required).
          attn_mask: the mask for the src sequence (optional).
          key_padding_mask: the mask for the src keys per batch (optional).
        Returns:
          src3: the output of transformer layer, share the same shape as src.
        """
        src2 = self.self_attn(
            self.norm(src), attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return src2
