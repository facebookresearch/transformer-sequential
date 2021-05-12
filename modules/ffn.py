#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardLayer(nn.Module):
    def __init__(self, args):
        super(FeedForwardLayer, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hid_sz, args.inner_hid_sz)
        self.fc2 = nn.Linear(args.inner_hid_sz, args.hid_sz)
        if self.args.pre_norm:
            self.fc2.weight.data.div_(math.sqrt(self.args.nlayers * 2))
            self.fc2.bias.data.div_(math.sqrt(self.args.nlayers * 2))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, h):
        h1 = self.fc1(h)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2
