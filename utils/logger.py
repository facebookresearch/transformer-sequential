#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class Logger(object):
    def __init__(self, args):
        self.args = args
        if args.plot:
            self.plot_path = os.path.join(args.plot_dir, args.plot_name)
            self.writer = SummaryWriter(self.plot_path)
        self.logs = dict()

    def print(self, msg):
        if self.args.local_rank == 0:
            print(msg)
        if self.args.plot:
            self.writer.add_text("stdout", msg)

    def set_state(self, state):
        self.logs = state

    def get_state(self):
        return self.logs

    def log(self, title, value):
        if title not in self.logs:
            self.logs[title] = {"data": [], "type": "line"}
        self.logs[title]["data"].append(value)

    def plot_step(self, step):
        for title, v in self.logs.items():
            if v["type"] == "line":
                if title == "X":
                    pass
                else:
                    self.writer.add_scalar(
                        title, v["data"][step], self.logs["X"]["data"][step]
                    )

    def plot_line(self, title, vals, X=None):
        if torch.is_tensor(vals):
            vals = vals.detach().cpu().numpy()
        if X is None:
            X = range(len(vals))
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        fig = plt.figure()
        plt.plot(vals)
        self.writer.add_figure(title, fig)

    def plot_bar(self, title, vals, X=None):
        if torch.is_tensor(vals):
            vals = vals.detach().cpu().numpy()
        if X is None:
            X = range(len(vals))
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        fig = plt.figure()
        plt.bar(X, vals)
        self.writer.add_figure(title, fig)

    def plot_heatmap(self, title, vals):
        if torch.is_tensor(vals):
            vals = vals.detach().cpu().numpy()
        fig = plt.figure()
        plt.imshow(vals, cmap="hot", interpolation="nearest")
        self.writer.add_figure(title, fig)

    def step(self, args, stat_train, stat_val, elapsed, gpu_mem):
        if "err" in stat_train:
            print(
                "{}\ttrain: {:.2f}%\tval: {:.2f}%\tms/batch: {:.1f}\tgpu_mem: {:.1f}gb".format(
                    (args.ep + 1) * args.nbatches // args.update_freq,
                    stat_train["err"] * 100,
                    stat_val["err"] * 100,
                    elapsed,
                    gpu_mem,
                )
            )
            self.log("loss/train", stat_train["loss"])
            self.log("loss/val", stat_val["loss"])
            self.log("err/train", stat_train["err"])
            self.log("err/val", stat_val["err"])
        elif args.data_type == "char":
            print(
                "{}\ttrain: {:.2f}bpc\tval: {:.2f}bpc\tms/batch: {:.1f}\tgpu_mem: {:.1f}gb".format(
                    (args.ep + 1) * args.nbatches // args.update_freq,
                    stat_train["loss"] / math.log(2),
                    stat_val["loss"] / math.log(2),
                    elapsed,
                    gpu_mem,
                )
            )
            self.log("loss/train", stat_train["loss"] / math.log(2))
            self.log("loss/val", stat_val["loss"] / math.log(2))
        else:
            train_ppl = math.exp(min(stat_train["loss"], 30))  # avoid overflow
            val_ppl = math.exp(min(stat_val["loss"], 30))  # avoid overflow
            print(
                "{}\ttrain_ppl: {:.1f}\tval_ppl: {:.1f}\tms/batch: {:.1f}\tgpu_mem: {:.1f}gb".format(
                    (args.ep + 1) * args.nbatches // args.update_freq,
                    train_ppl,
                    val_ppl,
                    elapsed,
                    gpu_mem,
                )
            )
            self.log("loss/train", stat_train["loss"])
            self.log("loss/val", stat_val["loss"])
            self.log("loss/ppl_train", train_ppl)
            self.log("loss/ppl_val", val_ppl)
        self.log("X", (args.ep + 1) * args.nbatches // args.update_freq)

        if args.plot:
            self.log("compute/gpu_mem_gb", gpu_mem)
            self.log("compute/batch_time_ms", elapsed)
            self.plot_step(-1)
            self.writer.flush()
