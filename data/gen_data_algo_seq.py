#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random

VARIABLE_NAMES = ["x", "y", "z", "p", "r"]
OPERATIONS = ["=", "++", "--", "if", "print"]
OPERATION_WEIGHTS = [1, 1, 1, 1, 1]
MAX_VAL = 10
MIN_VAL = 1

TASK_LEN = 100
NTASKS_TRAIN = 10000
NTASKS_TEST = 1000


class GenerationFail(Exception):
    pass


def gen_new_var(args, state):
    new_vars = [v for v in VARIABLE_NAMES[:args.variables] if v not in state.keys()]
    if len(new_vars) == 0:
        raise GenerationFail
    var = random.choice(new_vars)
    return var


def gen_vars(state, k=1):
    if len(state) < k:
        raise GenerationFail
    vars = random.sample(state.keys(), k=k)
    return vars


def gen_statement(
    args,
    state,
    nested=False,
):
    output = None
    while True:
        try:
            op = random.choices(OPERATIONS, weights=OPERATION_WEIGHTS, k=1)[0]
            if op == "=":
                var = gen_new_var(args, state)
                state[var] = random.randint(MIN_VAL, MAX_VAL)
                statement = f"{var} = {state[var]} ;"
            else:
                if op == "++":
                    var = gen_vars(state, 1)[0]
                    if state[var] == MAX_VAL:
                        raise GenerationFail
                    state[var] += 1
                    statement = f"{var} ++ ;"
                elif op == "--":
                    var = gen_vars(state, 1)[0]
                    if state[var] == MIN_VAL:
                        raise GenerationFail
                    state[var] -= 1
                    statement = f"{var} -- ;"
                elif op == "print":
                    if nested:
                        raise GenerationFail
                    var = gen_vars(state, 1)[0]
                    statement = f"print {var} ;"
                    output = f"_ _ {state[var]}"
                elif op == "if":
                    if nested:
                        raise GenerationFail
                    if random.random() < 0.5:
                        var, compare_var = gen_vars(state, 2)
                        compare_to = state[compare_var]
                    else:
                        var = gen_vars(state, 1)[0]
                        compare_to = random.randint(MIN_VAL, MAX_VAL)
                        compare_var = compare_to
                    if random.random() < 0.5:
                        if_op = "<"
                        if_cond = state[var] < compare_to
                    else:
                        if_op = ">"
                        if_cond = state[var] > compare_to
                    if if_cond:
                        sub_statement, _ = gen_statement(args, state, True)
                    else:
                        sub_statement, _ = gen_statement(args, state.copy(), True)
                    statement = f"if {var} {if_op} {compare_var} : {sub_statement}"
        except GenerationFail:
            # Try again.
            continue

        break

    if output is None:
        ntokens = len(statement.split())
        output = " ".join(["_"] * ntokens)

    return statement, output


def gen_task(args):
    N = args.task_len
    state = {}
    X, Y = [], []
    for i in range(N):
        x, y = gen_statement(args, state)
        X.append(x)
        Y.append(y)

    # Mark end of the task
    X += ["END"]
    Y += ["_"]
    X = " ".join(X)
    Y = " ".join(Y)
    return X, Y


def gen_data(args, path, ntasks):
    if os.path.exists(path):
        print(f"{path} already exists! Skipping generation.")
        return
    print(f"Generating {path}")

    X, Y = [], []
    for i in range(ntasks):
        x, y = gen_task(args)
        assert len(x.split()) == len(y.split())
        x += "\n"
        y += "\n"
        X.append(x)
        Y.append(y)

    fx = open(path, "w")
    fy = open(path + ".labels", "w")
    fx.writelines(X)
    fy.writelines(Y)
    fx.close()
    fy.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="")
    parser.add_argument("--variables", type=int, default=3, choices=(3, 5))
    parser.add_argument("--task-len", type=int, default=100)
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Creating directory: {args.path}")
        os.mkdir(args.path)
    gen_data(args, os.path.join(args.path, "train.txt"), NTASKS_TRAIN)
    gen_data(args, os.path.join(args.path, "valid.txt"), NTASKS_TEST)
    gen_data(args, os.path.join(args.path, "test.txt"), NTASKS_TEST)
