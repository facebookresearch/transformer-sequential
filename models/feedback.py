#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Original implementation of the Feedback Transformer
(see https://arxiv.org/abs/2002.09402v3).

A sequential transformer that exposes all previous representations
to all future representations, meaning the lowest representation of
the current timestep is formed from the highest-level abstract
representation of the past.

Size notations:
B = batch_sz
K = number of heads
H = hid_sz
M = mem_sz
L = the number of memories remaining
D = head size in some cases
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_seq import (
    TransformerSeq,
    SeqAttention,
    FeedForwardLayer,
)


def add_args(parser):
    parser.add_argument(
        "--share-proj-kv",
        action="store_true",
        default=False,
        help="share key value projections across layers",
    )


class BufferWrite(torch.autograd.Function):
    """
    Operation for incrementally writing to a buffer without creating a new
    memory.
    """

    @staticmethod
    def forward(ctx, input, buffer, pos, buffer_grad, windows):
        # assert input.size()[1:] == buffer.size()[1:]
        sz = input.size(0)
        buffer[pos : pos + sz] = input
        state = torch.tensor([pos, sz])  # bit faster on CPU
        ctx.save_for_backward(state, buffer_grad)
        out_list = []
        for w in windows:
            out = torch.zeros(
                1, device=input.device
            )  # a hacky way to keep _verion constant
            out.set_(buffer[pos + sz - w : pos + sz])
            out_list.append(out)
        return tuple(out_list)

    @staticmethod
    def backward(ctx, *grad_output_list):
        state, buffer_grad = ctx.saved_tensors
        pos = state[0].item()
        sz = state[1].item()
        for grad_output in grad_output_list:
            w = grad_output.size(0)
            buffer_grad[pos + sz - w : pos + sz] += grad_output
        grad_input = buffer_grad[pos : pos + sz]
        return grad_input, None, None, None, None


class SlidingWindowBuffer(object):
    """
    Fixed-length FIFO buffer with memory efficient backward.
    """

    def __init__(self, init_data, buffer_size, windows):
        super(SlidingWindowBuffer, self).__init__()
        # self.window_size = init_data.size()[0]
        self.windows = windows
        self.buffer_size = buffer_size
        # assert self.buffer_size >= self.window_size
        self.buffer_shape = (self.buffer_size,) + init_data.size()[1:]
        self.buffer = init_data.new_zeros(self.buffer_shape)
        self.buffer_grad = init_data.new_zeros(self.buffer_shape)
        self.buffer_pos = 0
        self.add(init_data)

    def add(self, x):
        # assert x.size()[1:] == self.buffer_shape[1:]
        # assert self.buffer_pos.item() + x.size(0) <= self.buffer_size
        self.out = BufferWrite.apply(
            x, self.buffer, self.buffer_pos, self.buffer_grad, self.windows
        )
        self.buffer_pos = self.buffer_pos + x.size(0)

    def get(self):
        # assert self.buffer_pos.item() >= self.window_size
        return self.out


class SelfAttnSerial(SeqAttention):
    """
    Self-attention optimized for serial compute.
    """

    def __init__(self, args):
        super(SelfAttnSerial, self).__init__(args)
        assert not self.args.adapt_span_cache  # unnecessary

    def prepare_adapt_span(self):
        if self.args.adapt_span:
            # compute adaptive-span mask once per block for efficiency
            _, _, key_pe, val_pe = self.adaptive_span.trim_memory(
                None, None, self.key_pe, self.val_pe
            )
            if key_pe is not None:
                self.key_pe_trimmed = key_pe.squeeze(0)
            if val_pe is not None:
                self.val_pe_trimmed = val_pe.squeeze(0)
            trim_len = self.adaptive_span.get_trim_len()
            self.adaptive_span.mask.prepare_mask(self.args.attn_lim - trim_len)
        else:
            if self.key_pe is not None:
                self.key_pe_trimmed = self.key_pe.squeeze(0)
            if self.val_pe is not None:
                self.val_pe_trimmed = self.val_pe.squeeze(0)

    def forward(self, query, key, value):
        # query = B x H
        # key, value = B x L x H

        attn = 0
        attn = attn + torch.bmm(key, query.unsqueeze(2)).squeeze(2)  # B x L
        attn = attn + torch.mm(query, self.key_pe_trimmed)  # B x L

        attn = attn / math.sqrt(self.args.head_dim)  # B X L
        attn = F.softmax(attn, dim=-1)
        if self.args.adapt_span:
            attn = self.adaptive_span(attn)
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        attn = self.dropout(attn)  # B X L

        out = 0
        out = out + torch.bmm(attn.unsqueeze(1), value).squeeze(1)  # B x H
        return out


class MultiHeadAttnSerial(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttnSerial, self).__init__()
        self.args = args
        self.attn = SelfAttnSerial(args)

        # init query and output projections
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

    def forward(self, query, key, value):
        query = self.proj_query(query)
        query = query.view(-1, self.args.head_dim)  # B_K x D
        # key and value are computed beforehand

        out = self.attn(query, key, value)  # B_K x D
        out = out.view(-1, 1, self.args.head_dim * self.args.nheads)  # B x K_D
        out = self.proj_out(out)
        return out


class TransformerLayerSerial(nn.Module):
    def __init__(self, args, layer_ind):
        super(TransformerLayerSerial, self).__init__()
        self.args = args
        self.attn = MultiHeadAttnSerial(args)
        self.ff = FeedForwardLayer(args)
        self.norm1 = nn.LayerNorm(args.hid_sz)
        self.norm2 = nn.LayerNorm(args.hid_sz)

    def forward(self, h, key, val):
        # h = B x 1 x H
        # h_prev = B x L x H
        if self.args.pre_norm:
            # assume key,val are already normalized
            attn_out = self.attn(self.norm1(h), key, val)
            h2 = h + attn_out  # B x M x H
            ff_out = self.ff(self.norm2(h2))
            out = h2 + ff_out  # B x M x H
        else:
            attn_out = self.attn(h, key, val)
            h2 = self.norm1(h + attn_out)  # B x M x H
            ff_out = self.ff(h2)
            out = self.norm2(h2 + ff_out)  # B x M x H
        return out

    def get_cache_size(self):
        return self.args.attn_lim


class FeedbackTransformer(TransformerSeq):
    def __init__(self, args):
        super(FeedbackTransformer, self).__init__(args)

        merged_layer_count = args.nlayers + 1
        self.single_memory_attn_params = nn.Parameter(
            torch.zeros(1, merged_layer_count)
        )
        self.register_buffer(
            "single_memory_attn_buf", torch.zeros(1, merged_layer_count)
        )

        if self.args.share_proj_kv:
            for l in range(1, len(self.layers)):
                self.get_layer(l).attn.proj_key.weight = self.get_layer(
                    0
                ).attn.proj_key.weight
                self.get_layer(l).attn.proj_val.weight = self.get_layer(
                    0
                ).attn.proj_val.weight
                if self.args.pre_norm:
                    # make sure key and values are normalized in the same way
                    self.get_layer(l).norm1.weight = self.get_layer(0).norm1.weight
                    self.get_layer(l).norm1.bias = self.get_layer(0).norm1.bias

    def build_layers(self):
        self.layers = nn.ModuleList()
        for l in range(self.args.nlayers):
            self.layers.append(TransformerLayerSerial(self.args, l))

    def merge_single_memory(self, h_all):
        # merge all layers into one
        sz = h_all[0].size()
        h_all = torch.stack(h_all)
        h_all = h_all.view(h_all.size(0), -1)
        single_memory_attn = F.softmax(self.single_memory_attn_params, dim=-1)
        h_single = torch.mm(single_memory_attn, h_all)
        h_single = h_single.view(sz)
        h_all = [h_single for _ in range(self.args.nlayers)]
        self.single_memory_attn_buf = single_memory_attn.detach()
        return h_all

    def cache_initprocess(self, h_cache):
        # compute key and value vectors only once
        key_cache, val_cache = [], []
        for l in range(self.args.nlayers):
            if self.args.feedback:
                h = h_cache[0]  # M x B x H
            else:
                h = h_cache[l]  # M x B x H

            windows = [self.args.attn_lim]
            if self.args.adapt_span:
                if self.args.share_proj_kv:
                    # keys and values differing in their spans
                    windows = []
                    for ll in range(self.args.nlayers):
                        trim_len = self.get_layer(ll).attn.attn.adaptive_span.get_trim_len()
                        windows.append(self.args.attn_lim - trim_len)
                else:
                    # avoid unnecessary computation
                    trim_len = self.get_layer(l).attn.attn.adaptive_span.get_trim_len()
                    h = h[trim_len:]
                    windows = [self.args.attn_lim - trim_len]

            if self.args.pre_norm:
                h = self.get_layer(l).norm1(h)

            key = self.get_layer(l).attn.proj_key(h)  # M x B x H
            val = self.get_layer(l).attn.proj_val(h)  # M x B x H
            key = key.view(h.size(0), -1, self.args.head_dim)  # M x B_K x D
            val = val.view(h.size(0), -1, self.args.head_dim)  # M x B_K x D
            key = SlidingWindowBuffer(key, key.size(0) + self.args.mem_sz, windows)
            val = SlidingWindowBuffer(val, val.size(0) + self.args.mem_sz, windows)
            key_cache.append(key)
            val_cache.append(val)

            if self.args.share_proj_kv:
                # key, values are identical across layers
                break

        # keep the original cache because it will be used in future
        h_cache = {"key": key_cache, "val": val_cache, "hid_prev": h_cache}
        h_cache["hid"] = [[] for _ in range(self.args.nlayers)]
        return h_cache

    def cache_preprocess(self, h_cache, t):
        if self.args.share_proj_kv:
            k = h_cache["key"][0].get()
            v = h_cache["val"][0].get()
            if self.args.adapt_span:
                # keys and values differing in their spans
                key_cache = [x.transpose(0, 1) for x in k]
                val_cache = [x.transpose(0, 1) for x in v]
            else:
                # there is a single set of keys and values
                key_cache = [k[0].transpose(0, 1)] * self.args.nlayers
                val_cache = [v[0].transpose(0, 1)] * self.args.nlayers
        else:
            key_cache = [h.get()[0].transpose(0, 1) for h in h_cache["key"]]
            val_cache = [h.get()[0].transpose(0, 1) for h in h_cache["val"]]
        return key_cache, val_cache

    def cache_postprocess(self, h_cache, h_all, t):
        for l in range(self.args.nlayers):
            h = h_all[l]
            # Compute key and value from the current state beforehand and
            # put them in the cache to be used in future steps.
            if self.args.pre_norm:
                h = self.get_layer(l).norm1(h)

            key = self.get_layer(l).attn.proj_key(h).view(1, -1, self.args.head_dim)
            val = self.get_layer(l).attn.proj_val(h).view(1, -1, self.args.head_dim)
            h_cache["key"][l].add(key)
            h_cache["val"][l].add(val)
            if not (self.args.feedback and l > 0):
                h = h.squeeze(1).unsqueeze(0)
                h_cache["hid"][l].append(h)
            if self.args.share_proj_kv:
                # no need to compute other layers as they are identical
                break
        return h_cache

    def cache_finalprocess(self, h_cache):
        # only keep the hid and recompute key,val again.
        h_cache_next = []
        for l in range(self.args.nlayers):
            hid_all = torch.cat([h_cache["hid_prev"][l]] + h_cache["hid"][l], dim=0)
            h_cache_next.append(hid_all[-self.args.attn_lim :])
            if self.args.feedback:
                break  # there is only one memory
        h_cache = h_cache_next
        return h_cache

    def forward(self, x, h_cache, target=None):
        # x : B x M
        assert x.size(1) == self.args.mem_sz

        h0_block = self.in_emb(x)  # B x M x H

        h_cache = self.cache_initprocess(h_cache)

        for l in range(self.args.nlayers):
            self.get_layer(l).attn.attn.prepare_adapt_span()

        # Bring the temporal dimension to the front.
        h0_block = h0_block.transpose(0, 1).contiguous()
        h0_block = h0_block.unsqueeze(
            2
        )  # Add a dummy temporal dimension because the model expects that.
        h_out_block = []
        for t in range(self.args.mem_sz):
            key_cache, val_cache = self.cache_preprocess(h_cache, t)

            h_t = h0_block[t]
            h_t_all = []
            for l in range(self.args.nlayers):
                h_t_all.append(h_t)
                h_t = self.get_layer(l)(h_t, key_cache[l], val_cache[l])  # B x M x H
            h_t_out = h_t
            h_out_block.append(h_t_out)

            if self.args.feedback:
                h_t_all.append(h_t_out)
                h_t_all = self.merge_single_memory(h_t_all)

            h_cache = self.cache_postprocess(h_cache, h_t_all, t)

        h_out = torch.cat(h_out_block, dim=1)
        if self.args.pre_norm:
            h_out = self.out_norm(h_out)
        out = self.out(h_out, target)

        h_cache = self.cache_finalprocess(h_cache)

        return out, h_cache, 0

    def init_hid_cache(self, batch_sz):
        hid = []
        for l in range(self.args.nlayers):
            # the temporal dimension must be at front so not to create
            # a new tensor at each time step
            h = torch.zeros(self.args.attn_lim, batch_sz, self.args.hid_sz)
            hid.append(h.to(self.args.device))
            if self.args.feedback:
                break  # there is only one memory
        return hid


def log(args, model, logger, stat_train):
    if args.plot and args.feedback:
        logger.plot_bar("layer_attn", model.module.single_memory_attn_buf.view(-1))
