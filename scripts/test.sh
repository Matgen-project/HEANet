#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u test_hea.py \
    --model-dir 'models/hea' \
    --output-dir 'test/' \
    --data-path 'datasets/examples/test' \
    --radius 4.0
