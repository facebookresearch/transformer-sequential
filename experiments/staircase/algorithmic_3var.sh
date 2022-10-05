#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

data_path=./data/algo_seq_3var/

# Generate data if needed
python ./data/gen_data_algo_seq.py --variables 3 --path $data_path

# Number of GPUs available on your machine
ngpus=1

# create experiment folder
mkdir ./staircase

# Baseline
python -m torch.distributed.launch --nproc_per_node=$ngpus main.py \
    --nepochs 50 --nbatches 1000 --batch-sz 512 --test-batch-sz 32 \
    --data $data_path --data-omit-labels "_" \
    --nlayers 8 --hid-sz 256 --inner-hid-sz 1024 --mem-sz 64 \
    --lr 0.0001 --momentum 0 --dropout 0.2 --optim adam --lr-warmup 1000 \
    --attn-lim 100 --nheads 4 --grad-clip 0.1 --pre-norm

# Staircase Models
python -m torch.distributed.launch --nproc_per_node=1 main.py \
    --nepochs 50 --nbatches 2000 --data $data_path --data-omit-labels "_" \
    --hid-sz 256 --inner-hid-sz 1024 --mem-sz 128 --batch-sz 256 --nlayers 4 \
    --lr 0.0001 --momentum 0 --dropout 0.2 --optim adam --lr-warmup 1000 \
    --nheads 4 --grad-clip 0.1 --test-batch-sz 32 --checkpoint-freq 25 \
    --pre-norm --staircase --fix-staircase-size-forward 64 \
    --validation-staircase-size-forward 64 --attn-lim 128 --staircase-size 128 \
    --checkpoint ./staircase/staircase_model
