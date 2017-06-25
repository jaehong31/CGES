#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1

python main.py \
    --base_lr 0.05 \
    --gpu_fraction 0.94 \
    --lamb $2 \
    --cges "$3" \
