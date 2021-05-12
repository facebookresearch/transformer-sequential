#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Usage: bash scripts/test.sh /path/to/model.pt 8

# Number of GPUs available on your machine.
ngpus=8

model=$1
path=$(dirname $1)
bsz=$2
name=$(basename $path)
args=$(cat $path/args.txt)
python -m torch.distributed.launch --nproc_per_node=$ngpus main.py $args \
--checkpoint $model --full-test --test-batch-sz $bsz --batch-sz $ngpus