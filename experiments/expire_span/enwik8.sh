#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Reproduces the enwik8 numbers used in the paper.

# Download data if needed
bash data/get_enwik8.sh

# Number of GPUs available on your machine.
ngpus=8

# If you run out of GPU memory, use --split-batch.

# Directory where the model checkpoints will save
base_dir=~/checkpoints

# The model trained in this run is then sharpened with a smaller LR. This run should result in:
# val bpc 1.027
# test bpc 1.005
python -m torch.distributed.launch --nproc_per_node=$ngpus main.py \
    --nepochs 100 --nbatches 1000 --data data/enwik8 \
    --hid-sz 512 --inner-hid-sz 2048 --mem-sz 512 --batch-sz 512 --nlayers 12 \
    --lr 0.0007 --momentum 0 --dropout 0.3 --optim adam --lr-warmup 8000 \
    --attn-lim 16384 --nheads 8 --grad-clip 0.3 \
    --checkpoint-freq 25 --test-batch-sz 128 --expire-span --expire-span-loss 0.000001 \
    --expire-span-ramp 128 --expire-span-noisy \
    --checkpoint $base_dir/model.pt

# Continue training with a smaller LR. This should give:
# val bpc 1.014
# test bpc 0.994
python -m torch.distributed.launch --nproc_per_node=$ngpus main.py \
    --nepochs 110 --nbatches 1000 --data data/enwik8 \
    --hid-sz 512 --inner-hid-sz 2048 --mem-sz 512 --batch-sz 512 --nlayers 12 \
    --lr 0.00007 --momentum 0 --dropout 0.3 --optim adam --lr-warmup 8000 \
    --attn-lim 16384 --nheads 8 --grad-clip 0.3 \
    --checkpoint-freq 25 --test-batch-sz 128 --expire-span --expire-span-loss 0.000001 \
    --expire-span-ramp 128 --expire-span-noisy \
    --checkpoint $base_dir/model.pt
