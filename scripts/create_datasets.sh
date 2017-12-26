#!/usr/bin/env bash

DATASETS_DIR=/var/tmp/datasets
rm -fr $DATASETS_DIR

cd `dirname $0`/..

python3 -m mult999.datasets \
  --filename-train $DATASETS_DIR/mult999.train \
  --filename-eval $DATASETS_DIR/mult999.eval
