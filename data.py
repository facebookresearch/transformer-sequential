#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from typing import List

import distributed


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).contiguous()
    return data


class Dictionary(object):
    def __init__(self):
        self.UNK = "<unk>"
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = []
        self.add_word(self.UNK)

    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 0

    def add_count(self, word):
        self.add_word(word)
        self.word2count[word] += 1

    def build_indices(self):
        sorted_dict = sorted(self.word2count.items(), key=lambda kv: kv[1])[::-1]
        for i in range(len(sorted_dict)):
            word = sorted_dict[i][0]
            self.word2idx[word] = i
            self.idx2word.append(word)

    @staticmethod
    def _split_line(line: str) -> List[str]:
        return line.split()

    def build(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = type(self)._split_line(line) + ["<eos>"]
                for word in words:
                    self.add_count(word)

        if os.path.exists(path + ".labels"):
            with open(path + ".labels", "r", encoding="utf8") as f:
                for line in f:
                    words = line.split() + ["<eos>"]
                    for word in words:
                        self.add_count(word)
        # Sort dictionary by count and build indices accordingly:
        self.build_indices()
        # self.__check__()

    def getidx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return self.word2idx[self.UNK]

    def __len__(self):
        return len(self.idx2word)

    def __check__(self):
        for i in range(min(26, self.__len__())):
            word = self.idx2word[i]
            print(i, word, self.word2count[word])


class CharDictionary(Dictionary):
    @staticmethod
    def _split_line(line: str) -> List[str]:
        return [c for c in line]


class Corpus(object):
    def __init__(self, path, include_eos=False):
        self.include_eos = include_eos
        self.dictionary = self._make_dictionary()

        print("building dictionary")
        self.dictionary.build(os.path.join(path, "train.txt"))

        print("tokenizing dataset")
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

        if os.path.exists(os.path.join(path, "train.txt.labels")):
            self.train_labels = self.tokenize(os.path.join(path, "train.txt.labels"))
            self.valid_labels = self.tokenize(os.path.join(path, "valid.txt.labels"))
            self.test_labels = self.tokenize(os.path.join(path, "test.txt.labels"))

    def _make_dictionary(self):
        return Dictionary()

    def _split_line(self, line):
        return line.split()

    def tokenize(self, path):
        print("tokenizing " + path)
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = self._split_line(line)
                if self.include_eos:
                    words += ["<eos>"]
                tokens += len(words)
        ids = torch.IntTensor(tokens)
        with open(path, "r", encoding="utf8") as f:
            token = 0
            for line in f:
                words = self._split_line(line)
                if self.include_eos:
                    words += ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.getidx(word)
                    token += 1
        return ids


class CharCorpus(Corpus):
    def _make_dictionary(self):
        return CharDictionary()

    def _split_line(self, line):
        return [c for c in line]


def get_data(
    args, logger, include_eos: bool = False
):
    corpus_path = os.path.join(args.data, "corpus.pt")
    if os.path.exists(corpus_path):
        corpus = torch.load(corpus_path)
        if include_eos:
            assert corpus.include_eos
    else:
        corpus = Corpus(args.data, include_eos)
        torch.save(corpus, corpus_path)

    args.vocab_sz = len(corpus.dictionary)
    logger.print(
        "Dictionary contains %d words (including the unk token)" % args.vocab_sz
    )

    train_data = batchify(corpus.train, args.batch_sz)
    val_data = batchify(corpus.valid, args.test_batch_sz)
    test_data = batchify(corpus.test, args.test_batch_sz)

    batch_sz_orig = args.batch_sz
    test_batch_sz_orig = args.test_batch_sz
    train_data, val_data, test_data = distributed.split_data(
        args, train_data, val_data, test_data
    )

    # don't move all data to device if we're going to move it as-needed later
    if not args.lazy_load_data:
        train_data = train_data.to(args.device)
        val_data = val_data.to(args.device)
        test_data = test_data.to(args.device)

    logger.print(
        "data len: train={} val={} test={}".format(
            train_data.size(1),
            val_data.size(1),
            test_data.size(1),
        )
    )

    if hasattr(corpus, "train_labels"):
        args.batch_sz = batch_sz_orig
        args.test_batch_sz = test_batch_sz_orig
        train_labels = batchify(corpus.train_labels, args.batch_sz).to(args.device)
        valid_labels = batchify(corpus.valid_labels, args.test_batch_sz).to(args.device)
        test_labels = batchify(corpus.test_labels, args.test_batch_sz).to(args.device)
        train_labels, valid_labels, test_labels = distributed.split_data(
            args, train_labels, valid_labels, test_labels
        )

        train_data = (train_data, train_labels)
        val_data = (val_data, valid_labels)
        test_data = (test_data, test_labels)

    return train_data, val_data, test_data, corpus


def reshape_batches(args, corpus):
    train_data = batchify(corpus.train, args.batch_sz)
    val_data = batchify(corpus.valid, args.test_batch_sz)
    test_data = batchify(corpus.test, args.test_batch_sz)

    train_data, val_data, test_data = distributed.split_data(
        args, train_data, val_data, test_data
    )
    return train_data, val_data, test_data
