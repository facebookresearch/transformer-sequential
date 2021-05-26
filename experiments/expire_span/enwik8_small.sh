#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This trains a small model on enwik8.

# Download data if needed
bash data/get_enwik8.sh

# This model uses 10-20gb of memory, so it's possible to train on 1 GPU.
# If you run out of GPU memory, use --split-batch 2.

# Directory where the model checkpoints will save
base_dir=~/checkpoints

python main.py \
    --nepochs 100 --nbatches 1000 --data data/enwik8 \
    --hid-sz 256 --inner-hid-sz 1024 --mem-sz 256 --batch-sz 64 --nlayers 8 \
    --lr 0.0003 --momentum 0 --dropout 0 --optim adam --lr-warmup 8000 \
    --attn-lim 4096 --nheads 4 --grad-clip 0.3 \
    --checkpoint-freq 25 --expire-span --expire-span-loss 0.000001 \
    --expire-span-ramp 64 --expire-span-pre-div 64 \
    --checkpoint $base_dir/enwik8_small.pt
