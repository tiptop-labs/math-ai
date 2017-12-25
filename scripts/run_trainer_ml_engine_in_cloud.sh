#!/usr/bin/env bash

TRAINER=mult999.first.nn0.task

NOW=$(date +"%Y%m%d%H%M%S")
JOB_ID=job_$NOW

BUCKET_NAME=math-ai
DATASETS_DIR=gs://$BUCKET_NAME/datasets
JOB_DIR=gs://$BUCKET_NAME/job_$NOW

gsutil rm $JOB_DIR** >> /dev/null 2>&1

cd `dirname $0`/..

gcloud ml-engine jobs submit training $JOB_ID \
  --job-dir $JOB_DIR\
  --module-name $TRAINER \
  --package-path mult999 \
  --config config.yaml \
  -- \
  --filename-train $DATASETS_DIR/mult999.train \
  --filename-eval $DATASETS_DIR/mult999.eval
