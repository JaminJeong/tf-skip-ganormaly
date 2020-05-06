#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    -i ~/.keras/datasets/flower_photos/daisy/9158041313_7a6a102f7a_n.jpg \
    -c ./training_checkpoints/ckpt-8 \