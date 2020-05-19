#!/usr/bin/env bash

declare sampling_method="smote"
declare -a list_k_neighbors=(3 5 7)
declare ratio_by_label="1=5.0"
for k_neighbors in ${list_k_neighbors[@]}; do
    python3 oversample.py \
        --data-path "./Datasets/Credit/train.csv" \
        --sampling-method "${sampling_method}" \
        --ratio-by-label "${ratio_by_label}" \
        --smote-k-neighbors ${k_neighbors} \
        --save-path "./Samples/Credit/${sampling_method}/k_neighbors=${k_neighbors}/ratio_by_label=${ratio_by_label}/sample_by_label.pkl" \
        --rand-seed 777
done

declare sampling_method="smote_svm"
declare -a list_k_neighbors=(5 7)
declare -a list_svm_kernel=("linear" "poly" "rbf" "sigmoid")
declare ratio_by_label="1=5.0"
for k_neighbors in ${list_k_neighbors[@]}; do
    for svm_kernel in ${list_svm_kernel[@]}; do
        python3 oversample.py \
            --data-path "./Datasets/Credit/train.csv" \
            --sampling-method "${sampling_method}" \
            --ratio-by-label "${ratio_by_label}" \
            --smote-k-neighbors ${k_neighbors} \
            --smote-svm-kernel ${svm_kernel} \
            --save-path "./Samples/Credit/${sampling_method}/k_neighbors=${k_neighbors}/svm_kernel=${svm_kernel}/ratio_by_label=${ratio_by_label}/sample_by_label.pkl" \
            --rand-seed 777
    done
done

declare sampling_method="gan"
declare -a list_size_latent=(100 120)
declare -a list_num_hidden_layers=(2 4)
declare ratio_by_label="1=5.0"
for size_latent in ${list_size_latent[@]}; do
    for num_hidden_layers in ${list_num_hidden_layers[@]}; do
        python3 oversample.py \
            --data-path "./Datasets/Credit/train.csv" \
            --sampling-method "${sampling_method}" \
            --ratio-by-label "${ratio_by_label}" \
            --gan-size-latent ${size_latent} \
            --gan-num-hidden-layers ${num_hidden_layers} \
            --save-path "./Samples/Credit/${sampling_method}/size_latent=${size_latent}/num_hidden_layers=${num_hidden_layers}/ratio_by_label=${ratio_by_label}/sample_by_label.pkl" \
            --rand-seed 777
    done
done
