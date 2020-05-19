#!/usr/bin/env bash

python3 search_GA.py \
    --train-data-path "./Datasets/Credit/train.csv" \
    --samples-dir "./Samples" \
    --ratio-min 2.0 \
    --ratio-max 5.0 \
    --ga-population-size 100 \
    --ga-crossover-size 2 \
    --ga-mutation-rate 0.001 \
    --ga-num-generations 100 \
    --classifier-num-hidden-layers 2 \
    --classifier-learning-rate 0.0001 \
    --classifier-beta-1 0.9 \
    --classifier-beta-2 0.999 \
    --classifier-batch-size 1024 \
    --classifier-num-epochs 100 \
    --classifier-run-device "cpu" \
    --rand-seed 777 \
    --verbose
