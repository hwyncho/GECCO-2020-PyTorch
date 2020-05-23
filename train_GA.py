#!/usr/bin/env python3
"""Train a DNN classifiers with sampled data."""
import argparse
import glob
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch

from classifiers import DNNClassifier
from classifiers import save_model
from classifiers import train


def load_data_with_samples(data_path: str, samples_dir: str, individual: list) -> tuple:
    """
    Function to load the csv data file.

    Parameters
    ----------
    data_path: str
    samples_dir: str
    individual: list

    Returns
    -------
    tuple

    """
    assert isinstance(data_path, str) and os.path.exists(data_path)
    assert isinstance(samples_dir, str) and os.path.exists(samples_dir)
    assert isinstance(individual, list)

    assert (os.path.splitext(data_path)[1]).lower() == ".csv"

    # Load the dataset.
    data_df: pd.DataFrame = pd.read_csv(data_path, delimiter=",")
    data_np: np.ndarray = data_df.values

    x_np: np.ndarray = data_np[:, 0:-1]
    y_np: np.ndarray = data_np[:, -1]

    list_sample_file: list = glob.glob(os.path.join(samples_dir, "**", "*.pkl"), recursive=True)
    list_sample_file.sort()

    list_sample_by_label: list = list()
    for sample_file in list_sample_file:
        with open(os.path.join(sample_file), mode="rb") as fp:
            list_sample_by_label.append(pickle.load(fp))

    y_stats: dict = Counter(y_np)
    list_label: list = list(list_sample_by_label[0].keys())

    for (i, ratio_by_label) in enumerate(individual):
        label: int = list_label[i]
        for (j, ratio_by_method) in enumerate(ratio_by_label):
            method: int = j
            if ratio_by_method > 0.0:
                number: int = int(y_stats[label] * ratio_by_method)
                new_x: np.ndarray = list_sample_by_label[method][label][0][:number]
                new_y: np.ndarray = list_sample_by_label[method][label][1][:number]

                x_np = np.concatenate([x_np, new_x], axis=0)
                y_np = np.concatenate([y_np, new_y], axis=0)

    x_tensor: torch.Tensor = torch.as_tensor(x_np, dtype=torch.float)
    y_tensor: torch.Tensor = torch.as_tensor(y_np, dtype=torch.long)

    return x_tensor, y_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for train a classifiers.")
    parser.add_argument("--train-data-path", type=str, required=True,
                        help="File path of a train data.",
                        dest="train_data_path")
    parser.add_argument("--samples-dir", type=str, default=None, required=False,
                        help="Directory path of store samples.",
                        dest="samples_dir")
    parser.add_argument("--model-save-path", type=str, required=True,
                        help="File path to store a trained classifiers model.",
                        dest="model_save_path")
    parser.add_argument("--num-hidden-layers", type=int, default=1, required=False,
                        help="Parameter num_hidden_layers of classifier.",
                        dest="num_hidden_layers")
    parser.add_argument("--batch-size", type=int, default=16, required=False,
                        help="Batch size during training.",
                        dest="batch_size")
    parser.add_argument("--num-epochs", type=int, default=1, required=False,
                        help="Number of training epochs.",
                        dest="num_epochs")
    parser.add_argument("--run-device", type=str, default="cpu", required=False,
                        help="Running device for PyTorch.",
                        dest="run_device")
    parser.add_argument("--learning-rate", type=float, default=0.001, required=False,
                        help="Learning rate for Adam optimizer.",
                        dest="learning_rate")
    parser.add_argument("--beta-1", type=float, default=0.9, required=False,
                        help="Beta 1 for Adam optimizer.",
                        dest="beta_1")
    parser.add_argument("--beta-2", type=float, default=0.999, required=False,
                        help="Beta 2 for Adam optimizer.",
                        dest="beta_2")
    parser.add_argument("--rand-seed", type=int, default=0, required=False,
                        help="Seed for generating random numbers.",
                        dest="rand_seed")
    parser.add_argument("--verbose", action='store_true', required=False,
                        help="Verbose")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    TRAIN_DATA_PATH: str = args.train_data_path
    SAMPLES_DIR: str = args.samples_dir
    MODEL_PATH: str = args.model_save_path
    NUM_HIDDEN_LAYERS: int = args.num_hidden_layers
    BATCH_SIZE: int = args.batch_size
    NUM_EPOCHS: int = args.num_epochs
    RUN_DEVICE: str = args.run_device
    LEARNING_RATE: float = args.learning_rate
    BETA_1: float = args.beta_1
    BETA_2: float = args.beta_2
    RAND_SEED: int = args.rand_seed
    VERBOSE: bool = args.verbose

    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(TRAIN_DATA_PATH)

    if not os.path.exists(SAMPLES_DIR):
        raise FileNotFoundError(SAMPLES_DIR)

    if os.path.exists(MODEL_PATH):
        raise FileExistsError(MODEL_PATH)

    assert isinstance(NUM_HIDDEN_LAYERS, int) and (NUM_HIDDEN_LAYERS > 0)
    assert isinstance(BATCH_SIZE, int) and (BATCH_SIZE > 0)
    assert isinstance(NUM_EPOCHS, int) and (NUM_EPOCHS > 0)
    assert isinstance(RUN_DEVICE, str) and (RUN_DEVICE.lower() in ["cpu", "cuda"])
    assert isinstance(LEARNING_RATE, float) and (LEARNING_RATE > 0.0)
    assert isinstance(BETA_1, float) and (0.0 <= BETA_1 < 1.0)
    assert isinstance(BETA_2, float) and (0.0 <= BETA_2 < 1.0)
    assert isinstance(RAND_SEED, int) and (RAND_SEED >= 0)
    assert isinstance(VERBOSE, bool)

    np.random.seed(seed=RAND_SEED)
    torch.manual_seed(seed=RAND_SEED)

    individual: list = [[]]

    x, y = load_data_with_samples(data_path=TRAIN_DATA_PATH,
                                  samples_dir=SAMPLES_DIR,
                                  individual=individual)
    size_features: int = x.size(1)
    size_labels: int = int(y.max().item() - y.min().item()) + 1

    # Train a classifiers and save the classifiers.
    classifier = DNNClassifier(size_features=size_features,
                               num_hidden_layers=NUM_HIDDEN_LAYERS,
                               size_labels=size_labels)
    trained_classifier = train(classifier=classifier,
                               x=x,
                               y=y,
                               batch_size=BATCH_SIZE,
                               num_epochs=NUM_EPOCHS,
                               run_device=RUN_DEVICE,
                               learning_rate=LEARNING_RATE,
                               beta_1=BETA_1,
                               beta_2=BETA_2,
                               rand_seed=RAND_SEED,
                               verbose=VERBOSE)
    save_model(classifier=trained_classifier, model_path=MODEL_PATH)

    print(">> Save the trained classifier: {0}".format((MODEL_PATH)))
