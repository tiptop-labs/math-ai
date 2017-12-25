#!/usr/bin/env bash

BUCKET_NAME=math-ai
DATASETS_DIR=gs://$BUCKET_NAME/datasets
gsutil rm $DATASETS_DIR** >> /dev/null 2>&1

cd `dirname $0`/..

python -m mult999.datasets \
  --filename-train $DATASETS_DIR/mult999.train \
  --filename-eval $DATASETS_DIR/mult999.eval
