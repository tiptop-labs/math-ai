#!/usr/bin/env bash

TRAINER=mult999.first.nn0.task

DATASETS_DIR=/var/tmp/datasets

JOB_DIR=/var/tmp/job
rm -fr $JOB_DIR

cd `dirname $0`/..

python3 -m $TRAINER \
  --job-dir $JOB_DIR \
  --filename-train $DATASETS_DIR/mult999.train \
  --filename-eval $DATASETS_DIR/mult999.eval \
  --summaries 1 \
  --profiling 1
