#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Transformer model for sequential prediction.

Size notations:
B = batch_sz
H = hid_sz
M = mem_sz
L = attn_lim
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import pos_emb, skew, unskew
from modules import AdaptiveSpan, FeedForwardLayer


def add_args(parser):
    parser.add_argument("--attn-lim", type=int, default=64, help="limit attention span")
    parser.add_argument(
        "--head-dim", type=int, default=0, help="head dimension. set automatically if 0"
    )
    parser.add_argument(
        "--pre-norm",
        action="store_true",
        default=False,
        help="apply layernorms to inputs to sublayers",
    )


class SeqAttention(nn.Module):
    """
    Sequential self-attention layer.

    Each position only attends to its previous L positions (doesn't include the current
    position) using relative position embeddings.
    """

    def __init__(self, args):
        super(SeqAttention, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        self.key_pe, self.val_pe = None, None
        self.key_pe = pos_emb(args, (1, args.head_dim, args.attn_lim))

        if self.args.adapt_span:
            self.adaptive_span = AdaptiveSpan(
                args,
                args.attn_lim,
                args.adapt_span_loss,
                args.adapt_span_len,
                args.adapt_span_init,
            )

    def forward(self, query, key, value):
        # query = B x M x H
        # key, value = B x (M+L) x H
        aux_loss = 0

        key_pe, val_pe = self.key_pe, self.val_pe
        if self.args.adapt_span:
            key, value, key_pe, val_pe = self.adaptive_span.trim_memory(
                key, value, key_pe, val_pe
            )

        attn = 0

        # compute attention from context
        attn = torch.matmul(
            query, key.transpose(-1, -2)
        )  # B x M (dest) x (M+L) (src)
        attn = unskew(attn)  # B x M x L

        # compute the effect of position embedding
        attn = attn + torch.matmul(query, key_pe)  # B x M x L

        attn = attn / math.sqrt(self.args.head_dim)  # B x M X L
        attn = F.softmax(attn, dim=-1)
        if self.args.adapt_span:
            attn = self.adaptive_span(attn)
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        attn = self.dropout(attn)  # B x M X L

        out = 0
        attn_cont = skew(attn, 0)  # B x M X (L+M)
        out = out + torch.matmul(attn_cont, value)  # B x M x H

        return out, aux_loss


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadSeqAttention, self).__init__()
        self.args = args
        self.attn = SeqAttention(args)

        self.proj_query = nn.Linear(
            args.hid_sz, args.head_dim * args.nheads, bias=False
        )
        self.proj_out = nn.Linear(args.head_dim * args.nheads, args.hid_sz, bias=False)
        if self.args.pre_norm:
            self.proj_out.weight.data.div_(math.sqrt(self.args.nlayers * 2))
        self.proj_val = nn.Linear(
            args.hid_sz, args.head_dim * args.nheads, bias=False
        )
        self.proj_key = nn.Linear(
            args.hid_sz, args.head_dim * args.nheads, bias=False
        )

    def head_reshape(self, x):
        K = self.args.nheads
        D = self.args.head_dim
        sz = x.size()
        sz = sz[:-1] + (K, D)  # B x (M+L) x K x D
        x = x.view(sz)  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value):
        B = query.size(0)
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out, aux_loss = self.attn(query, key, value)  # B_K x M x D
        out = out.view(B, self.args.nheads, M, self.args.head_dim)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)  # B x M x H
        return out, aux_loss


class TransformerSeqLayer(nn.Module):
    def __init__(self, args, layer_ind):
        super(TransformerSeqLayer, self).__init__()
        self.args = args
        self.attn = MultiHeadSeqAttention(args)
        self.ff = FeedForwardLayer(args)
        self.norm1 = nn.LayerNorm(args.hid_sz)
        self.norm2 = nn.LayerNorm(args.hid_sz)

    def forward(self, h, h_prev, **kargs):
        # h = B x M x H
        # h_prev = B x L x H
        h_memory = torch.cat([h_prev, h], dim=1)  # B x (L+M) x H

        if self.args.pre_norm:
            h_memory = self.norm1(h_memory)
            attn_out, aux_loss = self.attn(self.norm1(h), h_memory, h_memory)
        else:
            attn_out, aux_loss = self.attn(h, h_memory, h_memory)

        # FF
        if self.args.pre_norm:
            h2 = h + attn_out  # B x M x H
            ff_out = self.ff(self.norm2(h2))
            out = h2 + ff_out  # B x M x H
        else:
            h2 = self.norm1(h + attn_out)  # B x M x H
            ff_out = self.ff(h2)
            out = self.norm2(h2 + ff_out)  # B x M x H

        return out, aux_loss

    def get_cache_size(self):
        if self.args.adapt_span and self.args.adapt_span_cache:
            return self.attn.attn.adaptive_span.get_cache_size()
        else:
            return self.args.attn_lim


class TransformerOutput(nn.Module):
    def __init__(self, args):
        super(TransformerOutput, self).__init__()
        self.out_emb = nn.Linear(args.hid_sz, args.vocab_sz)

    def forward(self, x, target=None):
        return F.log_softmax(self.out_emb(x), dim=-1)


class TransformerSeq(nn.Module):
    def __init__(self, args):
        super(TransformerSeq, self).__init__()
        self.args = args

        self.in_emb = nn.Embedding(args.vocab_sz, args.hid_sz)

        self.build_layers()

        self.out = TransformerOutput(args)
        if self.args.pre_norm:
            self.out_norm = nn.LayerNorm(args.hid_sz)

        for l in range(1, len(self.layers)):
            self.layers[l].attn.attn.key_pe = self.layers[0].attn.attn.key_pe
            self.layers[l].attn.attn.val_pe = self.layers[0].attn.attn.val_pe

    def build_layers(self):
        self.layers = nn.ModuleList()
        for l in range(self.args.nlayers):
            self.layers.append(TransformerSeqLayer(self.args, l))

    def get_layer(self, layer_ind):
        return self.layers[layer_ind]

    def forward(self, x, h_prev, target=None):
        # x : B x M
        M = x.size(1)
        h = self.in_emb(x)  # B x M x H

        h_cache = []
        aux_loss = 0
        for li in range(self.args.nlayers):
            cache_size = self.get_layer(li).get_cache_size()
            if cache_size > M:
                h_cache.append(
                    torch.cat([h_prev[li][:, -cache_size + M :, :], h], dim=1)
                )
            else:
                h_cache.append(h[:, -cache_size:, :])
            h, al = self.get_layer(li)(h, h_prev[li])  # B x M x H
            aux_loss = aux_loss + al

        if self.args.pre_norm:
            h = self.out_norm(h)
        out = self.out(h, target)

        return out, h_cache, aux_loss

    def init_hid_cache(self, batch_sz):
        hid = []
        for l in range(self.args.nlayers):
            h = torch.zeros(
                batch_sz, self.get_layer(l).get_cache_size(), self.args.hid_sz
            )
            hid.append(h.to(self.args.device))
        return hid
