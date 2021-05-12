#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Compresssive Transformer (mean-pool version)

https://arxiv.org/abs/1911.05507

Size notations:
B = batch_sz
H = hid_sz
M = mem_sz
L = attn_lim
c = compression rate
C = compressed memory size (after compression)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import transformer_seq
from models.utils import pos_emb, skew, unskew
from modules import FeedForwardLayer


def add_args(parser):
    parser.add_argument("--compress-rate", type=int, default=4, help="compression rate")
    parser.add_argument(
        "--compress-size", type=int, default=128, help="memory size to be compressed"
    )


def unskew_step(X, step):
    B, M, _ = X.size()  # B x M x (M/c+C)
    X = X.view(B, M // step, step, -1)  # B x M/c x c x (M/c+C)
    X = X.transpose(1, 2).flatten(0, 1)  # Bc x M/c x (M/c+C)

    # unskew discards the last memory, but we need it here.
    # (this make two memories overlap a bit, otherwise there will be a gap)
    X = F.pad(X, [0, 1])  # Bc x M/c x (M/c+C+1)

    X = unskew(X)  # Bc x M/c x C+1
    X = X.view(B, step, M // step, -1)  # B x c x M/c x C+1
    X = X.transpose(1, 2).flatten(1, 2)  # B x M x C+1
    return X


def skew_step(X, step, pad_value):
    B, M, _ = X.size()  # B x M x C+1
    X = X.view(B, M // step, step, -1)  # B x M/c x c x C+1
    X = X.transpose(1, 2).flatten(0, 1)  # Bc x M/c x C+1
    X = skew(X, pad_value)  # Bc x M/c x M/c+C+1

    # remove the last column of zeros (because of added extra memory)
    X = X[:, :, :-1]  # Bc x M/c x M/c+C

    X = X.view(B, step, M // step, -1)  # B x c x M/c x M/c+C
    X = X.transpose(1, 2).flatten(1, 2)  # B x M x M/c+C
    return X


class SeqAttention(nn.Module):
    def __init__(self, args):
        super(SeqAttention, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        assert self.args.compress_size % self.args.compress_rate == 0
        C = self.args.compress_size // self.args.compress_rate
        self.key_pe, self.val_pe = None, None
        self.key_pe = pos_emb(args, (1, args.head_dim, args.attn_lim + C + 1))

    def forward(self, query, key, value, ckey, cvalue):
        # query = B x M x H
        # key, value = B x (M+L) x H
        # ckey, cvalue = B x M/c+C x H
        aux_loss = 0
        B, M, _ = query.size()
        c = self.args.compress_rate
        C = self.args.compress_size // self.args.compress_rate
        assert M % c == 0

        attn = 0
        # compute attention from context
        attn = torch.matmul(
            query, key.transpose(-1, -2)
        )  # B x M (dest) x (M+L) (src)
        attn = unskew(attn)  # B x M x L

        # compressed memory attention
        cattn = torch.matmul(query, ckey.transpose(-1, -2))  # B x M x M/c+C
        # Note that there is 1 extra memory. This ensure that two memories
        # overlaps without any gap.
        cattn = unskew_step(cattn, c)  # B x M x C+1
        attn = torch.cat([cattn, attn], dim=-1)  # B x M x C+L+1

        # compute the effect of position embedding
        attn = attn + torch.matmul(query, self.key_pe)  # B x M x C+L+1

        attn = attn / math.sqrt(self.args.head_dim)  # B x M X C+L+1
        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)  # B x M X C+L+1

        out = 0

        # compressed memory output
        cattn = attn[:, :, :C+1]  # B x M x C+1
        attn = attn[:, :, C+1:]
        cattn = skew_step(cattn, c, 0)  # B x M x M/c+C
        out = out + torch.matmul(cattn, cvalue)  # B x M x H

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

    def forward(self, query, key, value, ckey, cvalue):
        B = query.size(0)
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        cvalue = self.proj_val(cvalue)
        cvalue = self.head_reshape(cvalue)
        key = self.proj_key(key)
        key = self.head_reshape(key)
        ckey = self.proj_key(ckey)
        ckey = self.head_reshape(ckey)

        out, aux_loss = self.attn(query, key, value, ckey, cvalue)  # B_K x M x D
        out = out.view(B, self.args.nheads, M, self.args.head_dim)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out, aux_loss


class TransformerSeqLayer(nn.Module):
    def __init__(self, args, layer_ind):
        super(TransformerSeqLayer, self).__init__()
        self.args = args
        self.attn = MultiHeadSeqAttention(args)
        self.ff = FeedForwardLayer(args)
        self.norm1 = nn.LayerNorm(args.hid_sz)
        self.norm2 = nn.LayerNorm(args.hid_sz)

    def forward(self, h, h_memory, c_memory):
        # h = B x M x H
        # h_memory = B x L+M x H

        if self.args.pre_norm:
            h_memory = self.norm1(h_memory)
            c_memory = self.norm1(c_memory)
            attn_out, aux_loss = self.attn(
                self.norm1(h), h_memory, h_memory, c_memory, c_memory
            )
        else:
            attn_out, aux_loss = self.attn(h, h_memory, h_memory, c_memory, c_memory)

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
        return self.args.attn_lim


class CompressiveTransformer(transformer_seq.TransformerSeq):
    def build_layers(self):
        self.layers = nn.ModuleList()
        for l in range(self.args.nlayers):
            self.layers.append(TransformerSeqLayer(self.args, l))

    def get_layer(self, layer_ind):
        return self.layers[layer_ind]

    def forward(self, x, h_prev, target=None):
        # x : B x M
        B, M = x.size()
        H = self.args.hid_sz
        c = self.args.compress_rate
        C = self.args.compress_size // self.args.compress_rate

        h = self.in_emb(x)  # B x M x H

        c_prev = h_prev[-self.args.nlayers:]
        h_prev = h_prev[:-self.args.nlayers]

        h_cache = []
        c_cache = []
        aux_loss = 0
        for l in range(self.args.nlayers):
            h_memory = torch.cat([h_prev[l], h], dim=1)  # B x L+M x H

            # compress (note! there is overlap between two memories)
            new_compress = h_memory[:, :M, :]  # B x M x H
            new_compress = new_compress.view(B, M // c, c, H)
            new_compress = new_compress.mean(2)  # B x M/c x H
            c_memory = torch.cat([c_prev[l], new_compress], dim=1)  # B x C+M/c x H

            h, l = self.get_layer(l)(h, h_memory, c_memory)  # B x M x H
            aux_loss = aux_loss + l

            h_cache.append(h_memory[:, -self.args.attn_lim:, :])
            c_cache.append(c_memory[:, -C:, :])

        if self.args.pre_norm:
            h = self.out_norm(h)
        out = self.out(h, target)

        h_cache.extend(c_cache)

        return out, h_cache, aux_loss

    def init_hid_cache(self, batch_sz):
        hid = []
        for l in range(self.args.nlayers):
            h = torch.zeros(
                batch_sz, self.get_layer(l).get_cache_size(), self.args.hid_sz
            )
            hid.append(h.to(self.args.device))

        C = self.args.compress_size // self.args.compress_rate
        for l in range(self.args.nlayers):
            h = torch.zeros(batch_sz, C, self.args.hid_sz)
            hid.append(h.to(self.args.device))

        return hid
