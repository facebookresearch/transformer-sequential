#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import torch


def load_path(args, path):
    print("loading from " + path)
    # the model is saved from gpu0 so we need to map it to CPU first
    return torch.load(path, map_location=lambda storage, loc: storage)


def load_checkpoint(args, path, model, optimizer, logger, scheduler):
    f = load_path(args, path)
    ep_init = f["epoch"]
    model.load_state_dict(f["model"])
    logger.set_state(f["logger"])
    optimizer.load_state_dict(f["optimizer"])
    if "scheduler_epoch" in f:
        scheduler.step(f["scheduler_epoch"])

    return ep_init


def load(args, model, optimizer, logger, scheduler):
    ep_init = 0
    if args.checkpoint != "" and os.path.exists(args.checkpoint):
        try:
            ep_init = load_checkpoint(
                args, args.checkpoint, model, optimizer, logger, scheduler
            )
        except Exception as e:
            print(f"load failed: {e}")
            # try the backup checkpoint
            if os.path.exists(args.checkpoint + ".bak"):
                try:
                    ep_init = load_checkpoint(
                        args,
                        args.checkpoint + ".bak",
                        model,
                        optimizer,
                        logger,
                        scheduler,
                    )
                except Exception as e:
                    print(f"backup load failed: {e}")
    return ep_init


def save(args, model, optimizer, logger, scheduler):
    if args.checkpoint != "":
        if os.path.exists(args.checkpoint):
            if (
                args.checkpoint_freq > 0
                and args.ep > 0
                and args.ep % args.checkpoint_freq == 0
            ):
                try:
                    shutil.copyfile(
                        args.checkpoint, args.checkpoint + "." + str(args.ep)
                    )
                except:
                    print("save copy failed")
            # make a backup in case this save fails
            os.replace(args.checkpoint, args.checkpoint + ".bak")
        f = dict()
        f["epoch"] = args.ep + 1
        f["model"] = model.state_dict()
        f["logger"] = logger.get_state()
        f["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            f["scheduler_epoch"] = scheduler.last_epoch
        torch.save(f, args.checkpoint)
