#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

trainfile=$1

#RESULTDIR=/home/lichao/catedisp/third_No4/67_cilipian
RESULTDIR=$2
#DATADIR=/home/lichao/catedisp/third_No4/67_cilipian
DATADIR=$3

var=`echo $4 | awk -F',' '{print $0}' | sed "s/,/ /g"`
for i in $var
do
 echo $i
done

# fasttext supervised -input "${DATADIR}/$trainfile.train" -output "${RESULTDIR}/$trainfile" -dim 50 -lr 0.8 -lrUpdateRate 100 -ws 5 -wordNgrams 2 -minCount 1 -minCountLabel 1 -neg 5 -t 1e-5 -bucket 10000000 -epoch 30 -thread 32

# fasttext test "${RESULTDIR}/$trainfile.bin" "${DATADIR}/$trainfile.test"
# fasttext test "${RESULTDIR}/$trainfile.bin" "${DATADIR}/$trainfile.train"
