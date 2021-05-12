#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Implements Expire-Span (TODO: arxiv link).

For each hidden state, compute how long it can stay in memory. When a query
is beyond that limit, mask out the corresponding attention.

Size notations:
B'= batch_sz
K = number of heads
B = batch_sz x K
H = hid_sz
M = mem_sz
L = the number of memories remaining
"""

from __future__ import print_function
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import transformer_seq
from models.utils import skew, pos_emb
from modules import FeedForwardLayer


def add_args(parser):
    parser.add_argument(
        "--expire-span-loss",
        type=float,
        default=0,
        help="loss coefficient for reducing spans",
    )
    parser.add_argument(
        "--expire-span-ramp",
        type=int,
        default=32,
        help="ramp length of the masking function",
    )
    parser.add_argument(
        "--expire-span-init-percentage",
        type=float,
        default=0.1,
        help="sigmoid could center from a lower number than 0.5",
    )
    parser.add_argument(
        "--expire-span-noisy", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--expire-span-pre-div",
        type=float,
        default=1.0,
        help="divide activations before non-linearity",
    )
    parser.add_argument(
        "--expire-span-layerdrop",
        type=float,
        default=0.0,
        help="layer drop",
    )


def log(args, model, logger, stat_train):
    x = []
    x_max = 0
    for l in model.module.layers:
        if l.args.expire_span:
            s = l.attn.attn.expire_span.avg_span_log
            l.attn.attn.expire_span.avg_span_log = []
            x += s
            if hasattr(l.attn.attn.expire_span, "max_span_log"):
                x_max = max(x_max, l.attn.attn.expire_span.max_span_log)
                l.attn.attn.expire_span.max_span_log = 0
    if len(x) > 0:
        x = sum(x) / len(x)
        logger.log("adapt_span/avg", x)
        logger.log("adapt_span/max", x_max)


class ExpireSpanDrop(nn.Module):
    def __init__(self, args, size):
        super(ExpireSpanDrop, self).__init__()
        self.size = size
        self.args = args
        self.span_predictor = nn.Linear(self.args.hid_sz, 1)
        self.span_predictor.weight.data.fill_(0)
        b = -math.log((1.0 / args.expire_span_init_percentage - 1.0))
        self.span_predictor.bias.data.fill_(b)
        self.avg_span_log = []
        self.max_span_log = 0
        assert args.attn_lim >= args.mem_sz

    def forward(self, attn, memory_hid, current_counter):
        # Since we're dropping memories, here L can be smaller than attn_lim
        # attn : B x M x L
        # memory_hid : B' x L x H'
        # current_counter : B' x L
        B, M, L = attn.size()

        # Compute the maximum span (number of steps) a memory can stay
        max_span = self.span_predictor(
            memory_hid / self.args.expire_span_pre_div
        ).squeeze(
            -1
        )  # B' x L
        max_span = torch.sigmoid(max_span) * self.size

        if self.training:
            # Again, measure only for the current block.
            self.avg_span_log.append(max_span[:, -M:].mean().item())
            self.max_span_log = max(self.max_span_log, max_span[:, -M:].max().item())

        # Compute remaining spans measured from the 1st query.
        remaining_offset = max_span - current_counter  # B' x L

        # add noise
        if self.args.expire_span_noisy and self.training:
            noisy_span_lim = self.block_span_noise * self.size
            max_span_noisy = max_span.clamp(max=noisy_span_lim)
            remaining_offset_noisy = max_span_noisy - current_counter  # B' x L
        else:
            remaining_offset_noisy = remaining_offset

        # Remaining spans measured from all queries.
        remaining_span = remaining_offset_noisy.unsqueeze(1)  # B' x 1 x L
        remaining_span = remaining_span.expand(-1, M, -1).contiguous()  # B' x M x L
        remaining_span = remaining_span - torch.linspace(0, M - 1, M).view(1, -1, 1).to(
            device=remaining_span.device
        )

        # Compute the mask:
        #   mask=1 if remaining_span >= 0
        #   mask=0 if remaining_span <= -ramp_size
        #   In between, linearly interpolate between those two.
        mask = remaining_span / self.args.expire_span_ramp + 1.0
        mask = mask.clamp(0, 1)  # B' x M x L

        # Loss to encourage spans to be small.
        # Compute the loss for memories only under the ramp
        ramp_mask = (mask > 0) * (mask < 1)  # B' x M x L
        span_loss = remaining_span * ramp_mask.float()  # B' x M x L
        loss = span_loss.sum(dim=-1).sum(dim=-1)  # B'
        # Scale to match with previous versions:
        # - Divide by R because each memory has R losses applied
        # - Divide by M because we're avering over a block
        loss = loss / self.args.expire_span_ramp / M
        loss = loss * self.args.expire_span_loss  # B'

        # Replicate for each head.
        mask = mask.unsqueeze(1)  # B' x 1 x M x L
        mask = mask.expand(-1, self.args.nheads, -1, -1)  # B' x K x M x L
        mask = mask.flatten(0, 1)  # B x M x L

        return mask, loss, remaining_offset


class SeqAttention(nn.Module):
    def __init__(self, args):
        super(SeqAttention, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        self.key_pe, self.val_pe = None, None
        # Only last M steps will have position embeddings
        key_pe_size = min(args.attn_lim, args.mem_sz)
        self.key_pe = pos_emb(args, (1, args.head_dim, key_pe_size))

        max_mem_size = args.attn_lim + args.mem_sz

        if self.args.expire_span:
            self.expire_span = ExpireSpanDrop(args, self.args.attn_lim)
            max_mem_size += args.expire_span_ramp

        mask_causal = torch.ones(args.mem_sz, max_mem_size)  # M x L
        mask_causal = mask_causal.tril(diagonal=max_mem_size - args.mem_sz - 1)
        self.register_buffer("mask_causal", mask_causal)

    def forward(self, query, key, value, memory_hid, memory_counter):
        # query = B x M x H
        # key, value = B x L x H
        B, M, _ = query.size()
        _, L, _ = key.size()
        aux_loss = 0
        key_pe, val_pe = self.key_pe, self.val_pe
        spans = None

        # compute attention from context
        attn = torch.matmul(query, key.transpose(-1, -2))  # B x M x L
        # Since some memories are dropped, we cannot switch relative aligment
        # anymore. So we need work on absolute position alignment.

        # Mask out expired memories
        if self.args.expire_span:
            mask, expire_loss, spans = self.expire_span(
                attn, memory_hid, memory_counter
            )
            aux_loss = aux_loss + expire_loss
        else:
            mask = 1.0

        # Mask out attention to future steps (and t -> t)
        mask = mask * self.mask_causal[:, -L:]

        # Compute the effect of position embedding
        # Assume no memory is dropped from the previous block.
        attn_pos = torch.matmul(query, key_pe)  # B x M x L
        attn_pos = skew(attn_pos, 0)
        attn[:, :, -2 * M :] += attn_pos

        # Pre-softmax masking with -inf
        mask_pre = torch.zeros_like(attn).masked_fill_(mask.eq(0), float("-inf"))
        attn = attn + mask_pre

        attn = attn / math.sqrt(self.args.head_dim)  # B x M X L
        attn = F.softmax(attn, dim=-1)
        attn = attn * mask
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)

        attn = self.dropout(attn)  # B x M X L

        out = torch.matmul(attn, value)  # B x M x H

        return out, aux_loss, spans


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

    def forward(self, query, key, value, c_memory):
        B = query.size(0)
        M = query.size(1)

        assert key is value
        memory_hid = key

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out, aux_loss, spans = self.attn(
            query, key, value, memory_hid, c_memory
        )  # B_K x M x D
        out = out.view(B, self.args.nheads, M, self.args.head_dim)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out, aux_loss, spans


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
        # h_memory = B x L x H
        # c_memory = B x L

        if self.args.pre_norm:
            h_memory = self.norm1(h_memory)
            attn_out, aux_loss, spans = self.attn(
                self.norm1(h), h_memory, h_memory, c_memory
            )
        else:
            attn_out, aux_loss, spans = self.attn(h, h_memory, h_memory, c_memory)

        # FF
        if self.args.pre_norm:
            h2 = h + attn_out  # B x M x H
            ff_out = self.ff(self.norm2(h2))
            out = h2 + ff_out  # B x M x H
        else:
            h2 = self.norm1(h + attn_out)  # B x M x H
            ff_out = self.ff(h2)
            out = self.norm2(h2 + ff_out)  # B x M x H
        return out, aux_loss, spans

    def get_cache_size(self):
        return self.args.attn_lim


class ExpireSpan(transformer_seq.TransformerSeq):
    def build_layers(self):
        self.layers = nn.ModuleList()
        for l in range(self.args.nlayers):
            self.layers.append(TransformerSeqLayer(self.args, l))

    def forward(self, x, h_prev, target=None):
        # x : B x M
        M = x.size(1)
        B = x.size(0)
        H = self.args.hid_sz
        h = self.in_emb(x)  # B x M x H

        c_prev = h_prev[-self.args.nlayers :]
        h_prev = h_prev[: -self.args.nlayers]

        if self.args.expire_span_noisy and self.training:
            block_span_noise = random.random()
            for l in range(self.args.nlayers):
                if self.get_layer(l).args.expire_span:
                    self.get_layer(
                        l
                    ).attn.attn.expire_span.block_span_noise = block_span_noise

        h_cache = []  # memory including the current block
        c_cache = []  # the distance (in time steps) from the first query
        aux_loss = 0
        counter = torch.linspace(0, -M + 1, steps=M).to(self.args.device)
        counter = counter.view(1, -1).expand(B, -1)  # B x M
        for l in range(self.args.nlayers):
            h_cache.append(torch.cat([h_prev[l], h], dim=1))
            c_cache.append(torch.cat([c_prev[l], counter], dim=1))
            if self.training and self.args.expire_span_layerdrop > random.random():
                # skip this layer, but need to compute spans
                _, _, spans = self.get_layer(l)(h, h_cache[l], c_cache[l])  # B x M x H
            else:
                h, loss, spans = self.get_layer(l)(
                    h, h_cache[l], c_cache[l]
                )  # B x M x H
                aux_loss = aux_loss + loss
            if self.get_layer(l).args.expire_span:
                # Determine which memories can be dropped.
                # Extend spans by the ramp length R because memories are still
                # used during those R steps.
                spans = spans + self.args.expire_span_ramp  # B x L
                # Since spans are measured from the 1st query of this block,
                # subtract M so that they're measured from the next block.
                spans = spans - M
                # Now we can remove memories with span <= 0.
                spans = (spans > 0).float()

                # Do not drop any memory from the current block, so we can
                # compute relative-position embedding for last M steps easily.
                spans[:, -M:].fill_(1)

                # But because of batching, we need drop the same amount of memories.
                # Find out the smallest number of memories-to-drop within a batch.
                num_drop = (spans <= 0).long().sum(-1)
                num_drop_min = num_drop.min().item()
                # dropping arbitrary numbers might cause memory fragmentation,
                # so only drop with increments of mem_sz. Using mem_sz will
                # ensure that the memory size stay within the limit.
                num_drop_min = int(
                    math.floor(num_drop_min / self.args.mem_sz) * self.args.mem_sz
                )
                # Now only drop num_drop_min memories from each sample.
                # Here choose num_drop_min memories with the smallest span.
                #  minor speed ups, only sort when we want to drop
                if num_drop_min != 0:
                    spans_sorted, indices = spans.sort(dim=-1)
                    # from 0 to 1
                    spans_sorted[:, num_drop_min:] = 1
                    span_mask = torch.zeros_like(spans)
                    span_mask.scatter_(-1, indices, spans_sorted)
                    span_mask = span_mask.bool()
                    c_cache[l] = c_cache[l][span_mask].view(B, -1)  # B x L'
                    h_cache[l] = h_cache[l][span_mask].view(B, -1, H)  # B x L' x H
                # increase counter
                c_cache[l] += M
            else:
                attention_lim = self.get_layer(l).args.attn_lim
                # keep the nearest (L - M) tokens
                # B x (L x H)
                h_cache[l] = h_cache[l][:, -attention_lim:]  # B x L' x H
                c_cache[l] = c_cache[l][:, -attention_lim:]  # B x L'
                c_cache[l] += M

        if self.args.pre_norm:
            h = self.out_norm(h)
        out = self.out(h, target)

        h_cache.extend(c_cache)

        return out, h_cache, aux_loss

    def init_hid_cache(self, batch_sz):
        hid = []
        for l in range(self.args.nlayers):
            # It is enough to initialize the cache with M things
            init_cache_size = min(self.get_layer(l).args.attn_lim, self.args.mem_sz)
            h = torch.zeros(batch_sz, init_cache_size, self.args.hid_sz)
            hid.append(h.to(self.args.device))

        hid.extend(self.init_counter_cache(batch_sz))
        return hid

    def init_counter_cache(self, batch_sz):
        counter = []
        for l in range(self.args.nlayers):
            # It is enough to initialize the cache with M things
            h = torch.linspace(self.args.mem_sz, 1, steps=self.args.mem_sz)
            h = h.view(1, -1).expand(batch_sz, -1)
            counter.append(h.to(self.args.device))
        return counter
