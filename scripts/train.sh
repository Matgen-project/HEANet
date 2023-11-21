#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u main_hea.py \
    --output-dir 'models/examples' \
    --model-name 'graph_attention_transformer_oc20' \
    --input-irreps '100x0e' \
    --target 0 \
    --data-path 'datasets/examples' \
    --batch-size 16 \
    --radius 6.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --weight-decay 1e-3 \
    --lr 0.0002 \
    --min-lr 1e-6 \
    --energy-weight 1\
    --force-weight 0\
    --fold -1\
    --epochs 50
