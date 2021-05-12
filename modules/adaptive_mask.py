#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

# This class implements an adaptive masking function


class AdaptiveMask(nn.Module):
    def __init__(
        self,
        size,
        ramp_size,
        init_ratio=0,
        shape=(1,),
    ):
        super(AdaptiveMask, self).__init__()
        self.size = size
        self.ramp_size = ramp_size
        self.size_ratio = nn.Parameter(torch.zeros(*shape) + init_ratio)
        mask_template = torch.linspace(1 - size, 0, steps=size)
        self.register_buffer("mask_template", mask_template)

    def prepare_mask(self, span):
        mask = self.mask_template + self.size_ratio * self.size
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)
        if span < self.size:
            # the input could have been trimmed beforehand to save computation
            mask = mask.narrow(-1, self.size - span, span)
        self.mask_prepared = mask

    def forward(self, x):
        if hasattr(self, "mask_prepared"):
            return x * self.mask_prepared

        mask = self.mask_template + self.size_ratio * self.size
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self.size:
            # the input could have been trimmed beforehand to save computation
            mask = mask.narrow(-1, self.size - x.size(-1), x.size(-1))
        x = x * mask
        return x

    def get_max_size(self, include_ramp=True):
        max_size = self.size_ratio.max().item()
        max_size = max_size * self.size
        if include_ramp:
            max_size += self.ramp_size
        max_size = max(0, min(self.size, math.ceil(max_size)))
        return max_size

    def get_avg_size(self, include_ramp=True):
        avg_size = self.size_ratio.mean().item()
        avg_size = avg_size * self.size
        if include_ramp:
            avg_size += self.ramp_size
        avg_size = max(0, min(self.size, math.ceil(avg_size)))
        return avg_size

    def param_clamp(self):
        self.size_ratio.data.clamp_(0, 1)
