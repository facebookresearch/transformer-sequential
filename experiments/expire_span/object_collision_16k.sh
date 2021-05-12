#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

data_path=data/object_collisions/

# Generate data if needed
python data/gen_data_collisions.py --path $data_path

# Number of GPUs available on your machine
ngpus=8

python -m torch.distributed.launch --nproc_per_node=$ngpus main.py \
    --nepochs 100 --nbatches 1000 --batch-sz 512 --test-batch-sz 64 \
    --data $data_path --data-omit-labels 0 \
    --hid-sz 512 --inner-hid-sz 1024 --mem-sz 256 --nlayers 6 \
    --lr 0.0001 --momentum 0 --dropout 0.1 --optim adam --lr-warmup 1000 --grad-clip 0.1 \
    --attn-lim 16384 --nheads 4 --pre-norm \
    --expire-span --expire-span-loss 0.000002 --expire-span-init-percentage 0.01 --expire-span-ramp 128
