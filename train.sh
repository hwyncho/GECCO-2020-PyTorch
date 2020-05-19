#!/usr/bin/env bash

python3 train.py \
    --train-data-path "./Datasets/Credit/train.csv" \
    --model-save-path "./Trained-Classifiers/Credit/imbalanced/classifier.pth" \
    --num-hidden-layers 2 \
    --learning-rate 0.0001 \
    --beta-1 0.9 \
    --beta-2 0.999 \
    --batch-size 1024 \
    --num-epochs 100 \
    --run-device "cpu" \
    --rand-seed 777

declare sampling_method="smote"
declare -a list_k_neighbors=(3 5 7)
declare -a list_ratio_by_label=("1=2.0" "1=3.0" "1=4.0" "1=5.0")
for k_neighbors in ${list_k_neighbors[@]}; do
    for ratio_by_label in ${list_ratio_by_label[@]}; do
        python3 train.py \
            --train-data-path "./Datasets/Credit/train.csv" \
            --sampling-method "${sampling_method}" \
            --ratio-by-label "${ratio_by_label}" \
            --samples-dir "./Samples/Credit" \
            --smote-k-neighbors ${k_neighbors} \
            --model-save-path "./Trained-Classifiers/Credit/${sampling_method}/k_neighbors=${k_neighbors}/ratio_by_label=${ratio_by_label}/classifier.pth" \
            --num-hidden-layers 2 \
            --learning-rate 0.0001 \
            --beta-1 0.9 \
            --beta-2 0.999 \
            --batch-size 1024 \
            --num-epochs 100 \
            --run-device "cpu" \
            --rand-seed 777
    done
done

declare sampling_method="smote_svm"
declare -a list_k_neighbors=(5 7)
declare -a list_svm_kernel=("linear" "poly" "rbf" "sigmoid")
declare -a list_ratio_by_label=("1=2.0" "1=3.0" "1=4.0" "1=5.0")
for k_neighbors in ${list_k_neighbors[@]}; do
    for svm_kernel in ${list_svm_kernel[@]}; do
        for ratio_by_label in ${list_ratio_by_label[@]}; do
            python3 train.py \
                --train-data-path "./Datasets/Credit/train.csv" \
                --sampling-method "${sampling_method}" \
                --ratio-by-label "${ratio_by_label}" \
                --samples-dir "./Samples/Credit" \
                --smote-k-neighbors ${k_neighbors} \
                --smote-svm-kernel ${svm_kernel} \
                --model-save-path "./Trained-Classifiers/Credit/${sampling_method}/k_neighbors=${k_neighbors}/svm_kernel=${svm_kernel}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --num-hidden-layers 2 \
                --learning-rate 0.0001 \
                --beta-1 0.9 \
                --beta-2 0.999 \
                --batch-size 1024 \
                --num-epochs 100 \
                --run-device "cpu" \
                --rand-seed 777
        done
    done
done

declare sampling_method="gan"
declare -a list_size_latent=(100 120)
declare -a list_num_hidden_layers=(2 4)
declare -a list_ratio_by_label=("1=2.0" "1=3.0" "1=4.0" "1=5.0")
for size_latent in ${list_size_latent[@]}; do
    for num_hidden_layers in ${list_num_hidden_layers[@]}; do
        for ratio_by_label in ${list_ratio_by_label[@]}; do
            python3 train.py \
                --train-data-path "./Datasets/Credit/train.csv" \
                --sampling-method "${sampling_method}" \
                --ratio-by-label "${ratio_by_label}" \
                --samples-dir "./Samples/Credit" \
                --gan-size-latent ${size_latent} \
                --gan-num-hidden-layers ${num_hidden_layers} \
                --model-save-path "./Trained-Classifiers/Credit/${sampling_method}/size_latent=${size_latent}/num_hidden_layers=${num_hidden_layers}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --num-hidden-layers 2 \
                --learning-rate 0.0001 \
                --beta-1 0.9 \
                --beta-2 0.999 \
                --batch-size 1024 \
                --num-epochs 100 \
                --run-device "cpu" \
                --rand-seed 777
        done
    done
done
