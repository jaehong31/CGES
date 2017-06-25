#!/bin/sh
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

python main.py \
    --base_lr 0.05 \
    --gpu_fraction 0.94 \
    --lamb 0.00006 \
    --cges True \
