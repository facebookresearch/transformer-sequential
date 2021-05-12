#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def add_args(parser):
    parser.add_argument("--local_rank", type=int, default=0, help="")


def init(args):
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    args.rank = args.local_rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    if args.rank > 0:
        args.plot = False


def split_data(args, train_data, val_data, test_data):
    assert args.batch_sz % args.world_size == 0
    args.batch_sz = args.batch_sz // args.world_size
    train_data = train_data[args.batch_sz * args.rank : args.batch_sz * (args.rank + 1)]
    if args.test_batch_sz < args.world_size:
        # sometimes small test batch size is needed
        r = args.rank % args.test_batch_sz
        val_data = val_data[r : r + 1]
        test_data = test_data[r : r + 1]
        args.test_batch_sz = 1
    else:
        assert args.test_batch_sz % args.world_size == 0
        args.test_batch_sz = args.test_batch_sz // args.world_size
        val_data = val_data[
            args.test_batch_sz * args.rank : args.test_batch_sz * (args.rank + 1)
        ]
        test_data = test_data[
            args.test_batch_sz * args.rank : args.test_batch_sz * (args.rank + 1)
        ]
    return train_data, val_data, test_data


def wrap_model(args, model):
    model = model.to(args.device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
    )
    return model


def collect_stat(args, stat_train, stat_val, gpu_mem):
    X = torch.zeros(5).to(args.device)
    X[0] = stat_train["loss"]
    X[1] = stat_val["loss"]
    if "err" in stat_train:
        X[2] = stat_train["err"]
        X[3] = stat_val["err"]
    X[4] = gpu_mem
    torch.distributed.reduce(X, 0)
    torch.cuda.synchronize()
    if args.rank == 0:
        stat_train["loss"] = X[0].item() / args.world_size
        stat_val["loss"] = X[1].item() / args.world_size
        if "err" in stat_train:
            stat_train["err"] = X[2].item() / args.world_size
            stat_val["err"] = X[3].item() / args.world_size
        gpu_mem = X[4].item() / args.world_size
    return stat_train, stat_val, gpu_mem
