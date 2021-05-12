#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if [[ ! -d 'data/enwik8' ]]; then
  mkdir -p data/enwik8
  cd data/enwik8
  echo "Downloading enwik8 data ..."
  wget --continue http://mattmahoney.net/dc/enwik8.zip
  wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
  python3 prep_enwik8.py
  cd ../..
fi
