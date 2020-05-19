#!/usr/bin/env bash

python3 train_GA.py \
    --train-data-path "./Datasets/Credit/train.csv" \
    --samples-dir "./Samples" \
    --model-save-path "./Trained-Classifiers/Credit/ga/classifier.pth" \
    --num-hidden-layers 2 \
    --learning-rate 0.0001 \
    --beta-1 0.9 \
    --beta-2 0.999 \
    --batch-size 1024 \
    --num-epochs 16 \
    --run-device "cpu" \
    --rand-seed 777
