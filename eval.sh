#!/usr/bin/env bash

python3 eval.py \
    --eval-data-path "./Datasets/Credit/eval.csv" \
    --metric "f1_score" \
    --model-load-path "./Trained-Classifiers/Credit/imbalanced/classifier.pth" \
    --run-device "cpu"

declare sampling_method="smote"
declare -a list_k_neighbors=(3 5 7)
declare -a list_ratio_by_label=("1=2.0" "1=3.0" "1=4.0" "1=5.0")
for k_neighbors in ${list_k_neighbors[@]}; do
    for ratio_by_label in ${list_ratio_by_label[@]}; do
        python3 eval.py \
            --eval-data-path "./Datasets/Credit/eval.csv" \
            --metric "f1_score" \
            --model-load-path "./Trained-Classifiers/Credit/${sampling_method}/k_neighbors=${k_neighbors}/ratio_by_label=${ratio_by_label}/classifier.pth" \
            --run-device "cpu"
    done
done

declare sampling_method="smote_borderline"
declare -a list_k_neighbors=(5 7)
declare -a list_borderline_kind=("borderline-1" "borderline-2")
declare -a list_ratio_by_label=("1=2.0" "1=3.0" "1=4.0" "1=5.0")
for k_neighbors in ${list_k_neighbors[@]}; do
    for borderline_kind in ${list_borderline_kind[@]}; do
        for ratio_by_label in ${list_ratio_by_label[@]}; do
            python3 eval.py \
                --eval-data-path "./Datasets/Credit/eval.csv" \
                --metric "f1_score" \
                --model-load-path "./Trained-Classifiers/Credit/${sampling_method}/k_neighbors=${k_neighbors}/borderline_kind=${borderline_kind}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --run-device "cpu"
        done
    done
done

declare sampling_method="smote_svm"
declare -a list_k_neighbors=(5 7)
declare -a list_svm_kernel=("linear" "poly" "rbf" "sigmoid")
declare -a list_ratio_by_label=("1=2.0" "1=3.0" "1=4.0" "1=5.0")
for k_neighbors in ${list_k_neighbors[@]}; do
    for svm_kernel in ${list_svm_kernel[@]}; do
        for ratio_by_label in ${list_ratio_by_label[@]}; do
            python3 eval.py \
                --eval-data-path "./Datasets/Credit/eval.csv" \
                --metric "f1_score" \
                --model-load-path "./Trained-Classifiers/Credit/${sampling_method}/k_neighbors=${k_neighbors}/svm_kernel=${svm_kernel}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --run-device "cpu"
        done
    done
done

declare sampling_method="adasyn"
declare -a list_n_neighbors=(3 5 7)
declare -a list_ratio_by_label=("1=2.0" "1=3.0" "1=4.0" "1=5.0")
for n_neighbors in ${list_n_neighbors[@]}; do
    for ratio_by_label in ${list_ratio_by_label[@]}; do
        python3 eval.py \
            --eval-data-path "./Datasets/Credit/eval.csv" \
            --metric "f1_score" \
            --model-load-path "./Trained-Classifiers/Credit/${sampling_method}/n_neighbors=${n_neighbors}/ratio_by_label=${ratio_by_label}/classifier.pth" \
            --run-device "cpu"
    done
done

declare sampling_method="gan"
declare -a list_size_latent=(100 120)
declare -a list_num_hidden_layers=(2 4)
declare -a list_ratio_by_label=("1=2.0" "1=3.0" "1=4.0" "1=5.0")
for size_latent in ${list_size_latent[@]}; do
    for num_hidden_layers in ${list_num_hidden_layers[@]}; do
        for ratio_by_label in ${list_ratio_by_label[@]}; do
            python3 eval.py \
                --eval-data-path "./Datasets/Credit/eval.csv" \
                --metric "f1_score" \
                --model-load-path "./Trained-Classifiers/Credit/${sampling_method}/size_latent=${size_latent}/num_hidden_layers=${num_hidden_layers}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --run-device "cpu"
        done
    done
done

python3 eval.py \
    --eval-data-path "./Datasets/Credit/eval.csv" \
    --metric "f1_score" \
    --model-load-path "./Trained-Classifiers/Credit/ga/classifier.pth" \
    --run-device "cpu"
