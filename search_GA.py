#!/usr/bin/env python3
"""Run a GA."""
import argparse
import glob
import os
import pickle
from pprint import pprint

import numpy as np
import pandas as pd
import torch

import ga


def load_data(data_path: str) -> tuple:
    """
    Function to load the csv data file.

    Parameters
    ----------
    data_path: str

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


def load_samples(samples_dir: str) -> list:
    """
    Parameters
    ----------
    samples_dir: str

    Returns
    -------
    list

    """
    assert isinstance(samples_dir, str)

    if not os.path.exists(samples_dir):
        raise FileNotFoundError(samples_dir)

    list_sample_file: list = glob.glob(os.path.join(samples_dir, "**", "*.pkl"), recursive=True)
    list_sample_file.sort()

    sample_by_label: list = list()
    for sample_file in list_sample_file:
        with open(os.path.join(sample_file), mode="rb") as fp:
            sample_by_label.append(pickle.load(fp))

    return sample_by_label


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for Genetic Algorithm.")
    parser.add_argument("--train-data-path", type=str, required=True,
                        help="File path of a train data.",
                        dest="train_data_path")
    parser.add_argument("--samples-dir", type=str, default=None, required=True,
                        help="Directory path of store samples.",
                        dest="samples_dir")
    parser.add_argument("--ratio-min", type=float, default=1.0, required=False,
                        help="Minimum ratio for oversampling.",
                        dest="ratio_min")
    parser.add_argument("--ratio-max", type=float, default=1.1, required=False,
                        help="Maximum ratio for oversampling.",
                        dest="ratio_max")
    parser.add_argument("--ga-population-size", type=int, default=4, required=False,
                        help="Size of population. (GA)",
                        dest="ga_population_size")
    parser.add_argument("--ga-selection-method", type=str, default="roulette", required=False,
                        help="Method of selection. (GA)",
                        dest="ga_selection_method")
    parser.add_argument("--ga-crossover-method", type=str, default="onepoint", required=False,
                        help="Method of crossover. (GA)",
                        dest="ga_crossover_method")
    parser.add_argument("--ga-crossover-size", type=int, default=2, required=False,
                        help="Size of mated individual. (GA)",
                        dest="ga_crossover_size")
    parser.add_argument("--ga-mutation-method", type=str, default="swap", required=False,
                        help="Method of mutation. (GA)",
                        dest="ga_mutation_method")
    parser.add_argument("--ga-mutation-rate", type=float, default=0.01, required=False,
                        help="Ratio of mutation. (GA)",
                        dest="ga_mutation_rate")
    parser.add_argument("--ga-replacement-method", type=str, default="parents", required=False,
                        help="Method of replacement. (GA)",
                        dest="ga_replacement_method")
    parser.add_argument("--ga-num-generations", type=int, default=1, required=False,
                        help="Iteration of generation. (GA)",
                        dest="ga_num_generations")
    parser.add_argument("--ga-checkpoint-dir", type=str, default=None, required=False,
                        help="Directory path to store checkpoints. (GA)",
                        dest="ga_checkpoint_dir")
    parser.add_argument("--classifier-num-hidden-layers", type=int, default=1, required=False,
                        help="Number of hidden layers in classifier. (Classifier)",
                        dest="classifier_num_hidden_layers")
    parser.add_argument("--classifier-batch-size", type=int, default=16, required=False,
                        help="Batch size during training. (Classifier)",
                        dest="classifier_batch_size")
    parser.add_argument("--classifier-num-epochs", type=int, default=1, required=False,
                        help="Number of training epochs. (Classifier)",
                        dest="classifier_num_epochs")
    parser.add_argument("--classifier-run-device", type=str, default="cpu", required=False,
                        help="Running device for PyTorch. (Classifier)",
                        dest="classifier_run_device")
    parser.add_argument("--classifier-learning-rate", type=float, default=0.001, required=False,
                        help="Learning rate for Adam optimizer. (Classifier)",
                        dest="classifier_learning_rate")
    parser.add_argument("--classifier-beta-1", type=float, default=0.9, required=False,
                        help="Beta 1 for Adam optimizer. (Classifier)",
                        dest="classifier_beta_1")
    parser.add_argument("--classifier-beta-2", type=float, default=0.999, required=False,
                        help="Beta 2 for Adam optimizer. (Classifier)",
                        dest="classifier_beta_2")
    parser.add_argument("--rand-seed", type=int, default=0, required=False,
                        help="Seed for generating random numbers.",
                        dest="rand_seed")
    parser.add_argument("--verbose", action='store_true', required=False,
                        help="Verbose")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    TRAIN_DATA_PATH: str = args.train_data_path
    SAMPLES_DIR: str = args.samples_dir
    RATIO_MIN: float = args.ratio_min
    RATIO_MAX: float = args.ratio_max
    GA_POPULATION_SIZE: int = args.ga_population_size
    GA_SELECTION_METHOD: str = args.ga_selection_method
    GA_CROSSOVER_METHOD: str = args.ga_crossover_method
    GA_CROSSOVER_SIZE: int = args.ga_crossover_size
    GA_MUTATION_METHOD: str = args.ga_mutation_method
    GA_MUTATION_RATE: float = args.ga_mutation_rate
    GA_REPLACEMENT_METHOD: str = args.ga_replacement_method
    GA_NUM_GENERATIONS: int = args.ga_num_generations
    GA_CHECKPOINT_DIR: str = args.ga_checkpoint_dir
    CLASSIFIER_NUM_HIDDEN_LAYERS: int = args.classifier_num_hidden_layers
    CLASSIFIER_BATCH_SIZE: int = args.classifier_batch_size
    CLASSIFIER_NUM_EPOCHS: int = args.classifier_num_epochs
    CLASSIFIER_RUN_DEVICE: str = args.classifier_run_device
    CLASSIFIER_LEARNING_RATE: float = args.classifier_learning_rate
    CLASSIFIER_BETA_1: float = args.classifier_beta_1
    CLASSIFIER_BETA_2: float = args.classifier_beta_2
    RAND_SEED: int = args.rand_seed
    VERBOSE: bool = args.verbose

    assert isinstance(GA_POPULATION_SIZE, int) and (GA_POPULATION_SIZE > 0)
    assert isinstance(GA_SELECTION_METHOD, str)
    assert isinstance(GA_CROSSOVER_METHOD, str)
    assert isinstance(GA_CROSSOVER_SIZE, int) and (1 < GA_CROSSOVER_SIZE <= GA_POPULATION_SIZE)
    assert isinstance(GA_MUTATION_METHOD, str)
    assert isinstance(GA_MUTATION_RATE, float) and (0.0 <= GA_MUTATION_RATE <= 1.0)
    assert isinstance(GA_REPLACEMENT_METHOD, str)
    assert isinstance(GA_NUM_GENERATIONS, int) and (GA_NUM_GENERATIONS > 0)
    if GA_CHECKPOINT_DIR is not None:
        assert isinstance(GA_CHECKPOINT_DIR, str)
    assert isinstance(CLASSIFIER_NUM_HIDDEN_LAYERS, int) and (CLASSIFIER_NUM_HIDDEN_LAYERS > 0)
    assert isinstance(CLASSIFIER_BATCH_SIZE, int) and (CLASSIFIER_BATCH_SIZE > 0)
    assert isinstance(CLASSIFIER_NUM_EPOCHS, int) and (CLASSIFIER_NUM_EPOCHS > 0)
    assert isinstance(CLASSIFIER_RUN_DEVICE, str) and (CLASSIFIER_RUN_DEVICE.lower() in ["cpu", "cuda"])
    assert isinstance(CLASSIFIER_LEARNING_RATE, float) and (CLASSIFIER_LEARNING_RATE > 0.0)
    assert isinstance(CLASSIFIER_BETA_1, float) and (0.0 <= CLASSIFIER_BETA_1 < 1.0)
    assert isinstance(CLASSIFIER_BETA_2, float) and (0.0 <= CLASSIFIER_BETA_2 < 1.0)
    assert isinstance(RAND_SEED, int) and (RAND_SEED >= 0)
    assert isinstance(VERBOSE, bool)

    np.random.seed(seed=RAND_SEED)
    torch.manual_seed(seed=RAND_SEED)

    train_x, train_y = load_data(data_path=TRAIN_DATA_PATH)
    list_sample_by_label: list = load_samples(samples_dir=SAMPLES_DIR)

    size_features: int = train_x.size(1)
    size_labels: int = int(train_y.max().item() - train_y.min().item()) + 1

    population, logbook = ga.run(x=train_x,
                                 y=train_y,
                                 list_sample_by_label=list_sample_by_label,
                                 ratio_min=RATIO_MIN,
                                 ratio_max=RATIO_MAX,
                                 population_size=GA_POPULATION_SIZE,
                                 selection_method=GA_SELECTION_METHOD,
                                 crossover_method=GA_CROSSOVER_METHOD,
                                 crossover_size=GA_CROSSOVER_SIZE,
                                 mutation_method=GA_MUTATION_METHOD,
                                 mutation_rate=GA_MUTATION_RATE,
                                 replacement_method=GA_REPLACEMENT_METHOD,
                                 num_generations=GA_NUM_GENERATIONS,
                                 checkpoint_dir=GA_CHECKPOINT_DIR,
                                 rand_seed=RAND_SEED,
                                 verbose=VERBOSE,
                                 classifier_num_hidden_layers=CLASSIFIER_NUM_HIDDEN_LAYERS,
                                 classifier_batch_size=CLASSIFIER_BATCH_SIZE,
                                 classifier_num_epochs=CLASSIFIER_NUM_EPOCHS,
                                 classifier_run_device=CLASSIFIER_RUN_DEVICE,
                                 classifier_learning_rate=CLASSIFIER_LEARNING_RATE,
                                 classifier_beta_1=CLASSIFIER_BETA_1,
                                 classifier_beta_2=CLASSIFIER_BETA_2)

    if GA_CHECKPOINT_DIR is not None:
        with open(os.path.join(GA_CHECKPOINT_DIR, "logbook.pkl"), mode="wb") as fp:
            pickle.dump(logbook, fp)
