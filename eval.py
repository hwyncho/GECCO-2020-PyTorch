#!/usr/bin/env python3
"""Evaluate a trained DNN classifiers."""
import argparse
import os
from pprint import pprint

import numpy as np
import pandas as pd
import torch

from classifiers import DNNClassifier
from classifiers import load_model
from classifiers import evaluate


def load_dataset(data_path: str) -> tuple:
    """
    Function to load the csv data file.

    Parameters
    ----------
    dataset_path: str

    Returns
    -------
    tuple

    """
    assert isinstance(data_path, str)

    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    assert (os.path.splitext(data_path)[1]).lower() == '.csv'

    # Load the dataset.
    data_df: pd.DataFrame = pd.read_csv(data_path, delimiter=',')
    data_np: np.ndarray = data_df.values

    x_np: np.ndarray = data_np[:, 0:-1]
    y_np: np.ndarray = data_np[:, -1]

    x_tensor: torch.Tensor = torch.as_tensor(x_np, dtype=torch.float)
    y_tensor: torch.Tensor = torch.as_tensor(y_np, dtype=torch.long)

    return x_tensor, y_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for train a classifiers.")
    parser.add_argument("--eval-data-path", type=str, required=True,
                        help="File path of a eval data.",
                        dest="eval_data_path")
    parser.add_argument("--metric", type=str, default="f1_score", required=False,
                        help="Evaluation metrics.",
                        dest="metric")
    parser.add_argument("--model-load-path", type=str, required=True,
                        help="File path to store a trained classifiers model.",
                        dest="model_load_path")
    parser.add_argument("--run-device", type=str, default="cpu", required=False,
                        help="Running device for PyTorch.",
                        dest="run_device")
    parser.add_argument("--rand-seed", type=int, default=0, required=False,
                        help="Seed for generating random numbers.",
                        dest="rand_seed")
    parser.add_argument("--verbose", action='store_true', required=False,
                        help="Verbose")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    EVAL_DATASET_PATH: str = args.eval_data_path
    METRIC: str = args.metric
    MODEL_PATH: str = args.model_load_path
    RUN_DEVICE: str = args.run_device
    RAND_SEED: int = args.rand_seed
    VERBOSE: bool = args.verbose

    assert isinstance(METRIC, str) and (METRIC.lower() in ["f1_score", "confusion_matrix"])

    if not os.path.exists(EVAL_DATASET_PATH):
        raise FileNotFoundError(EVAL_DATASET_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    assert isinstance(RUN_DEVICE, str) and (RUN_DEVICE.lower() in ["cpu", "cuda"])
    assert isinstance(VERBOSE, bool)

    np.random.seed(seed=RAND_SEED)
    torch.manual_seed(seed=RAND_SEED)

    x, y = load_dataset(EVAL_DATASET_PATH)
    size_features: int = x.size(1)
    size_labels: int = int(y.max().item() - y.min().item()) + 1

    # Load the trained classifiers and evaluate the classifiers.
    classifier: torch.nn.Module = DNNClassifier(size_features=size_features,
                                                num_hidden_layers=2,
                                                size_labels=size_labels)
    trained_classifier: torch.nn.Module = load_model(classifier=classifier, model_path=MODEL_PATH)
    result: np.ndarray = evaluate(classifier=trained_classifier,
                                  x=x,
                                  y=y,
                                  metric=METRIC,
                                  run_device=RUN_DEVICE,
                                  rand_seed=RAND_SEED,
                                  verbose=VERBOSE)

    print(">> Evaluate the trained classifier: {0}".format(MODEL_PATH))
    if METRIC.lower() == "confusion_matrix":
        print(">> Confusion Matrix :")
        pprint(result)
        print("")
    elif METRIC.lower() == "f1_score":
        print(">> F1 score :")
        pprint(result)
        print("")
