#!/usr/bin/bash
# model_idx, max_epoch, train_batch_size, eval_batch_size
export CUDA_VISIBLE_DEVICES=3
python src/run.py 34 100 8 5