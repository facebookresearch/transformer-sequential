#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import copy
import math
import argparse
import random
import torch
import torch.optim as optim

import data
import distributed
from models import (
    compressive,
    expire_span,
    feedback,
    transformer_seq,
)
from modules import adaptive_span
from trainer import train
from utils.logger import Logger
import utils.checkpoint as checkpoint


def get_parser():
    parser = argparse.ArgumentParser()
    # model related
    parser.add_argument("--hid-sz", type=int, default=256, help="hidden size")
    parser.add_argument(
        "--inner-hid-sz", type=int, default=1024, help="inner hidden size of FF layer"
    )
    parser.add_argument("--nlayers", type=int, default=8, help="number of layers")
    parser.add_argument("--mem-sz", type=int, default=64, help="memory size")
    parser.add_argument(
        "--nheads", type=int, default=2, help="number of attention heads"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout rate of ReLU and attention"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="use the compressive transformer",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        default=False,
        help="use the feedback transformer, computing one step at a time like RNNs",
    )
    parser.add_argument(
        "--expire-span",
        action="store_true",
        default=False,
        help="compute expiration span for each memory",
    )
    # optimization related
    parser.add_argument("--lr", type=float, default=0.03, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--batch-sz", type=int, default=64, help="batch size")
    parser.add_argument(
        "--test-batch-sz",
        type=int,
        default=0,
        help="set different batch size for test and val data if greater than 0",
    )
    parser.add_argument(
        "--nbatches", type=int, default=1000, help="number of batches in each epoch"
    )
    parser.add_argument(
        "--nepochs", type=int, default=1000, help="number of epochs to train"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="sgd",
        choices=("sgd", "adam"),
        help="optimization method",
    )
    parser.add_argument(
        "--lr-warmup",
        type=int,
        default=0,
        help="linearly increase LR from 0 during K updates",
    )
    parser.add_argument(
        "--lr-decay",
        action="store_true",
        default=False,
        help="decay learning rate with cosine scheduler",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=0, help="clip gradient value",
    )
    parser.add_argument(
        "--split-batch",
        type=int,
        default=1,
        help="split batches into smaller pieces so it can fit in memory",
    )
    # data related
    parser.add_argument(
        "--data", type=str, help="data file location", required=True,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="char",
        choices=["char", "word"],
        help="data type",
    )
    parser.add_argument(
        "--data-eos",
        action="store_true",
        default=False,
        help="include the end-of-line as as token",
    )
    parser.add_argument(
        "--data-omit-labels",
        nargs="+",
        type=str,
        default=[],
        help="do not train on those labels",
    )
    # plotting
    parser.add_argument(
        "--plot", action="store_true", default=False, help="plot in tensorboard"
    )
    parser.add_argument(
        "--plot-dir", type=str, default="tensorboard_runs", help="tensorboard log dir",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="",
        help="tensorboard log name (default: datetime)",
    )
    # misc
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="path to save/load model"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=0, help="how often to keep a copy"
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        default=False,
        help="do testing on whole data",
    )
    parser.add_argument(
        "--full-valid",
        action="store_true",
        default=False,
        help="do validation on whole data (during training)",
    )
    parser.add_argument(
        "--lazy-load-data",
        action="store_true",
        default=False,
        help="moves data to GPU one sample at a time",
    )
    transformer_seq.add_args(parser)
    distributed.add_args(parser)
    adaptive_span.add_args(parser)
    compressive.add_args(parser)
    expire_span.add_args(parser)
    feedback.add_args(parser)
    return parser


def update_args(args):
    if args.head_dim == 0:
        assert args.hid_sz % args.nheads == 0
        args.head_dim = args.hid_sz // args.nheads

    args.update_freq = args.split_batch
    if args.split_batch > 1:
        assert args.batch_sz % args.split_batch == 0
        assert args.test_batch_sz % args.split_batch == 0
        args.batch_sz = args.batch_sz // args.split_batch
        args.test_batch_sz = args.test_batch_sz // args.split_batch
        args.nbatches *= args.split_batch
        args.lr_warmup *= args.split_batch

    if args.plot and args.plot_name == "":
        args.plot_name = time.strftime("%Y%m%d_%H%M%S")

    if args.test_batch_sz == 0:
        args.test_batch_sz = args.batch_sz


def main(args):
    args = copy.deepcopy(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    update_args(args)

    distributed.init(args)
    args.device = torch.device("cuda" if use_cuda else "cpu")
    logger = Logger(args)
    logger.print(f"PyTorch version: {torch.__version__}")
    logger.print(f"PyTorch CUDA version: {torch.version.cuda}")
    logger.print(str(args))

    # load data
    train_data, val_data, test_data, corpus = data.get_data(args, logger, args.data_eos)
    if len(args.data_omit_labels) > 0:
        args.data_omit_label_idx = [
            corpus.dictionary.word2idx[w] for w in args.data_omit_labels
        ]
    else:
        args.data_omit_label_idx = None

    # create a model
    if args.feedback:
        model = feedback.FeedbackTransformer(args)
    elif args.expire_span:
        model = expire_span.ExpireSpan(args)
    elif args.compress:
        model = compressive.CompressiveTransformer(args)
    else:
        model = transformer_seq.TransformerSeq(args)
    model.to(args.device)

    # count params
    nparameters = 0
    params = []
    for param in model.parameters():
        if param.requires_grad:
            nparameters += param.numel()
            params.append(param)
    logger.print("nparameters={:.2f}M".format(nparameters / 1e6))

    # OPTIM param
    if args.optim == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif args.optim == "adam":
        optimizer = optim.Adam(params, lr=args.lr)

    if args.lr_decay:
        # will do warm-up manually later
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.nepochs * args.nbatches
        )
    elif args.lr_warmup > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / args.lr_warmup)
        )
    else:
        scheduler = None

    model = distributed.wrap_model(args, model)

    ep_init = checkpoint.load(args, model, optimizer, logger, scheduler)

    # pos: data samling 0=sequential, -1=random
    pos = [0 for _ in range(3)]
    if isinstance(train_data, tuple):
        pos[0] = random.randrange(train_data[0].size(1) - args.mem_sz)
    else:
        pos[0] = random.randrange(train_data.size(1) - args.mem_sz)
    hid_cache = [
        model.module.init_hid_cache(args.batch_sz),
        model.module.init_hid_cache(args.test_batch_sz),
        model.module.init_hid_cache(args.test_batch_sz),
    ]

    if args.full_test:
        # perform evaluation only
        with torch.no_grad():
            stat_val, pos[1], hid_cache[1] = train(
                args,
                model,
                optimizer,
                scheduler,
                val_data,
                test_only=True,
                train_pos=pos[1],
                h_cache=hid_cache[1],
                corpus=corpus,
            )
            stat_test, pos[2], hid_cache[2] = train(
                args,
                model,
                optimizer,
                scheduler,
                test_data,
                test_only=True,
                train_pos=pos[2],
                h_cache=hid_cache[2],
                corpus=corpus,
            )
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            stat_test, stat_val, gpu_mem = distributed.collect_stat(
                args, stat_test, stat_val, gpu_mem
            )
            if args.data_type == "char":
                if "err" in stat_val:
                    logger.print("val err: {:.3f}%".format(stat_val["err"] * 100))
                    logger.print("test err: {:.3f}%".format(stat_test["err"] * 100))
                else:
                    logger.print(
                        "val: {:.3f}bpc".format(stat_val["loss"] / math.log(2))
                    )
                    logger.print(
                        "test: {:.3f}bpc".format(stat_test["loss"] / math.log(2))
                    )
            else:
                logger.print("val: {:.3f}ppl".format(math.exp(stat_val["loss"])))
                logger.print("test: {:.3f}ppl".format(math.exp(stat_test["loss"])))
            logger.print(f"gpu_mem: {gpu_mem:.1f}gb")
        return

    for ep in range(ep_init, args.nepochs):
        t_sta = time.time()
        args.ep = ep
        stat_train, pos[0], hid_cache[0] = train(
            args,
            model,
            optimizer,
            scheduler,
            train_data,
            train_pos=pos[0],
            h_cache=hid_cache[0],
            corpus=corpus,
        )
        elapsed = 1000 * (time.time() - t_sta) / args.nbatches
        with torch.no_grad():
            if args.full_valid:
                stat_val, _, _ = train(
                    args,
                    model,
                    optimizer,
                    scheduler,
                    val_data,
                    test_only=True,
                    train_pos=pos[1],
                    h_cache=hid_cache[1],
                    corpus=corpus,
                )
            else:
                stat_val, pos[1], hid_cache[1] = train(
                    args,
                    model,
                    optimizer,
                    scheduler,
                    val_data,
                    test_only=True,
                    train_pos=pos[1],
                    h_cache=hid_cache[1],
                    corpus=corpus,
                )

        gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        torch.cuda.reset_max_memory_allocated()
        stat_train, stat_val, gpu_mem = distributed.collect_stat(
            args, stat_train, stat_val, gpu_mem
        )

        if args.rank == 0:
            # only the master process will do logging, plotting and checkpoint
            if args.lr_decay:
                logger.log("compute/lr", optimizer.param_groups[0]["lr"])
            if args.adapt_span:
                adaptive_span.log(args, model, logger, stat_train)
            if args.expire_span:
                expire_span.log(args, model, logger, stat_train)
            if args.feedback:
                feedback.log(args, model, logger, stat_train)

            logger.step(args, stat_train, stat_val, elapsed, gpu_mem)
            checkpoint.save(args, model, optimizer, logger, scheduler)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
