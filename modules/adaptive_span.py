#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.adaptive_mask import AdaptiveMask

# Adaptive attention span for Transformer


def add_args(parser):
    parser.add_argument(
        "--adapt-span",
        action="store_true",
        default=False,
        help="enable adaptive attention span",
    )
    parser.add_argument(
        "--adapt-span-loss", type=float, default=0, help="loss coeff on attention span"
    )
    parser.add_argument(
        "--adapt-span-len", type=float, default=32, help="ramp length of adaptive span"
    )
    parser.add_argument(
        "--adapt-span-init", type=float, default=0, help="initial attention span ratio"
    )
    parser.add_argument(
        "--adapt-span-cache",
        action="store_true",
        default=False,
        help="adapt cache size to reduce memory usage",
    )
    parser.add_argument(
        "--adapt-span-trim-step", type=int, default=64, help="trim step"
    )
    parser.add_argument(
        "--adapt-span-layer",
        action="store_true",
        default=False,
        help="constrain all heads in a layer to have same span",
    )


def log(args, model, logger, stat_train):
    x = []
    for i, l in enumerate(model.module.layers):
        span = l.attn.attn.adaptive_span.mask.size_ratio.view(-1)
        x.append(span)
        span = span.mean().item()
    x = torch.cat(x, dim=0) * args.attn_lim
    logger.log("adapt_span/avg", x.mean().item())
    logger.log("adapt_span/max", x.max().item())
    if args.plot:
        logger.plot_bar("adapt_span/latest", x)


class AdaptiveSpan(nn.Module):
    def __init__(self, args, size, loss_coeff, ramp_size, init_ratio):
        super(AdaptiveSpan, self).__init__()
        self.size = size
        self.loss_coeff = loss_coeff
        self.args = args
        if self.args.adapt_span_layer:
            self.mask = AdaptiveMask(self.size, ramp_size, init_ratio=init_ratio)
        else:
            self.mask = AdaptiveMask(
                self.size, ramp_size, init_ratio=init_ratio, shape=(args.nheads, 1, 1),
            )

    def forward(self, attn):
        if self.args.adapt_span_layer:
            attn = self.mask(attn)
        elif self.args.feedback:
            B = attn.size(0)
            attn = attn.reshape(B // self.args.nheads, self.args.nheads, 1, -1)
            attn = self.mask(attn)
            attn = attn.view(B, -1)
        else:
            B, M = attn.size(0), attn.size(1)
            attn = attn.reshape(B // self.args.nheads, self.args.nheads, M, -1)
            attn = self.mask(attn)
            attn = attn.view(B, M, -1)
        return attn

    # how many steps can be skipped
    def get_trim_len(self):
        L = self.size
        trim_len = min(L - 1, L - self.mask.get_max_size())
        trim_len = (
            math.floor(trim_len / self.args.adapt_span_trim_step)
            * self.args.adapt_span_trim_step
        )  # for better memory caching
        return trim_len

    # determine how long the cache should be
    def get_cache_size(self):
        trim_len = self.get_trim_len()
        # give a buffer of 64 steps as spans can increase during training
        return min(self.size, self.size - trim_len + 64)

    # trim out unnecessary memory computation
    def trim_memory(self, key, value, key_pe, val_pe):
        trim_len = self.get_trim_len()
        if key is not None:
            if self.args.feedback:
                cache_size = key.size(1)
            else:
                cache_size = key.size(1) - self.args.mem_sz
            trim_len_cache = trim_len - (self.size - cache_size)
            if self.args.feedback:
                # keys and values must have cut to the right sizes beforehand.
                # Also adapt_span_cache=False, so cache can't be shorter.
                assert trim_len_cache == 0
            if trim_len_cache > 0:
                key = key[:, trim_len_cache:, :]
                value = value[:, trim_len_cache:, :]
            elif trim_len_cache < 0:
                print(
                    "warning: cache is too short. cache_size={} trim_len={}".format(
                        cache_size, trim_len
                    )
                )
                key = F.pad(key, [0, 0, -trim_len_cache, 0])
                value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
            if val_pe is not None:
                val_pe = val_pe[:, trim_len:, :]
        return key, value, key_pe, val_pe

    # compute the loss
    def get_loss(self):
        return self.mask.size_ratio.mean() * self.loss_coeff * self.size

    def param_clamp(self):
        self.mask.param_clamp()
