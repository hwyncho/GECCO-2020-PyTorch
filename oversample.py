#!/usr/bin/env python3
"""Oversample dataset with various methods."""
import argparse
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn.svm import SVC

from gan import Discriminator
from gan import Generator
from gan import predict as gan_predict
from gan import train as gan_train


def oversampling(data_path: str,
                 sampling_method: str,
                 ratio_by_label: dict,
                 rand_seed: int = 0,
                 **kwargs) -> dict:
    """
    Function to load the csv data file and over-sample the minority data.

    Parameters
    ----------
    data_path: str
    sampling_method: str
    ratio_by_label: dict
    rand_seed: int

    Returns
    -------
    dict

    """
    assert isinstance(data_path, str)
    assert isinstance(sampling_method, str)
    assert sampling_method.lower() in ["smote", "smote_svm", "gan"]
    assert isinstance(ratio_by_label, dict)
    assert isinstance(rand_seed, int) and rand_seed > 0

    sampling_method = sampling_method.lower()

    # Set the seed for generating random numbers.
    np.random.seed(seed=rand_seed)

    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    assert (os.path.splitext(data_path)[1]).lower() == ".csv"

    smote_k_neighbors: int = 5
    if "smote_k_neighbors" in kwargs:
        assert isinstance(kwargs["smote_k_neighbors"], int) and (kwargs["smote_k_neighbors"] > 0)
        smote_k_neighbors = kwargs["smote_k_neighbors"]

    smote_svm_kernel: str = "rbf"
    if "smote_svm_kernel" in kwargs:
        assert isinstance(kwargs["smote_svm_kernel"], str)
        smote_svm_kernel = str(kwargs["smote_svm_kernel"]).lower()

    gan_size_latent: int = 100
    if "gan_size_latent" in kwargs:
        assert isinstance(kwargs["gan_size_latent"], int) and (kwargs["gan_size_latent"] > 0)
        gan_size_latent = kwargs["gan_size_latent"]

    gan_num_hidden_layers: int = 1
    if "gan_num_hidden_layers" in kwargs:
        assert isinstance(kwargs["gan_num_hidden_layers"], int) and (kwargs["gan_num_hidden_layers"] >= 1)
        gan_num_hidden_layers = kwargs["gan_num_hidden_layers"]

    # Load the dataset.
    data_df: pd.DataFrame = pd.read_csv(data_path, delimiter=',')
    data_np: np.ndarray = data_df.values

    x: np.ndarray = data_np[:, 0:-1]
    y: np.ndarray = data_np[:, -1]

    y_stats: dict = Counter(y)
    number_by_label: dict = dict()
    sampling_strategy: dict = dict()
    for (label, ratio) in ratio_by_label.items():
        label, ratio = int(label), float(ratio)
        assert ratio > 1.0
        if label in y_stats:
            number_by_label[label] = int(y_stats[label] * (ratio - 1.0))
            sampling_strategy[label] = int(y_stats[label] * ratio)
        else:
            raise RuntimeError("{0} class is not in {1}.".format(label, data_path))

    if sampling_method == "smote":
        smote: SMOTE = SMOTE(sampling_strategy=sampling_strategy,
                             random_state=rand_seed,
                             k_neighbors=smote_k_neighbors)
        new_x, new_y = smote.fit_resample(x, y)
        new_x, new_y = new_x[len(x):], new_y[len(y):]
    elif sampling_method == "smote_svm":
        smote: SVMSMOTE = SVMSMOTE(sampling_strategy=sampling_strategy,
                                   random_state=rand_seed,
                                   k_neighbors=smote_k_neighbors,
                                   svm_estimator=SVC(kernel=smote_svm_kernel))
        new_x, new_y = smote.fit_resample(x, y)
        new_x, new_y = new_x[len(x):], new_y[len(y):]
    elif sampling_method == "gan":
        batch_size: int = 1024
        num_epochs: int = 100
        run_device: str = "cpu"
        if torch.cuda.is_available():
            run_device = "cuda"
        learning_rate: float = 0.0001
        beta_1: float = 0.9
        beta_2: float = 0.999

        generator: Generator = Generator(size_latent=gan_size_latent,
                                         size_labels=len(y_stats.keys()),
                                         num_hidden_layers=gan_num_hidden_layers,
                                         size_outputs=x.shape[1])
        discriminator: Generator = Discriminator(size_inputs=x.shape[1],
                                                 size_labels=len(y_stats.keys()),
                                                 num_hidden_layers=gan_num_hidden_layers)
        trained_G, trained_D = gan_train(generator=generator,
                                         discriminator=discriminator,
                                         x=torch.as_tensor(x, dtype=torch.float32),
                                         y=torch.as_tensor(y, dtype=torch.long),
                                         latent_size=gan_size_latent,
                                         batch_size=batch_size,
                                         num_epochs=num_epochs,
                                         run_device=run_device,
                                         learning_rate=learning_rate,
                                         beta_1=beta_1, beta_2=beta_2,
                                         rand_seed=rand_seed,
                                         verbose=False)
        new_x, new_y = gan_predict(generator=trained_G, discriminator=trained_D,
                                   latent_size=gan_size_latent, output_by_label=number_by_label,
                                   run_device=run_device,
                                   rand_seed=rand_seed)
    else:
        raise ValueError()

    sample_by_label: dict = dict()
    for (label, number) in number_by_label.items():
        sample_by_label[label] = (new_x[:number], new_y[:number])
        new_x, new_y = new_x[number:], new_y[number:]

    return sample_by_label


def parse_args():
    class StoreDictKeyPair(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            my_dict: dict = dict()
            for kv in values.split(","):
                k, v = kv.split("=")
                my_dict[k] = v
            setattr(namespace, self.dest, my_dict)

    parser = argparse.ArgumentParser(description="Arguments for oversample a dataset.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="File path of a data.",
                        dest="data_path")
    parser.add_argument("--sampling-method", type=str, default=None, required=True,
                        help="Name of method to oversample a dataset.",
                        dest="sampling_method")
    parser.add_argument("--ratio-by-label", action=StoreDictKeyPair, required=False,
                        metavar="KEY1=VAL1,KEY2=VAL2...",
                        help="Oversampling ratio by label.",
                        dest="ratio_by_label")
    parser.add_argument("--smote-k-neighbors", type=int, default=5, required=False,
                        help="Parameter k_neighbors of SMOTE.",
                        dest="smote_k_neighbors")
    parser.add_argument("--smote-svm-kernel", type=str, default="linear", required=False,
                        help="Parameter svm_kernel of SVMSMOTE.",
                        dest="smote_svm_kernel")
    parser.add_argument("--gan-size-latent", type=int, default=100, required=False,
                        help="Parameter size_latent of GAN.",
                        dest="gan_size_latent")
    parser.add_argument("--gan-num-hidden-layers", type=int, default=2, required=False,
                        help="Parameter num_hidden_layers of GAN.",
                        dest="gan_num_hidden_layers")
    parser.add_argument("--save-path", type=str, required=True,
                        help="File path to store oversampled data.",
                        dest="model_save_path")
    parser.add_argument("--rand-seed", type=int, default=0, required=False,
                        help="Seed for generating random numbers.",
                        dest="rand_seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DATA_PATH = args.data_path
    SAMPLING_METHOD: str = args.sampling_method
    RATIO_BY_LABEL: dict = args.ratio_by_label
    SMOTE_K_NEIGHBORS: int = args.smote_k_neighbors
    SMOTE_SVM_KERNEL: str = args.smote_svm_kernel
    GAN_SIZE_LATENT: int = args.gan_size_latent
    GAN_NUM_HIDDEN_LAYERS: int = args.gan_num_hidden_layers
    SAVE_PATH: str = args.model_save_path
    RAND_SEED: int = args.rand_seed

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    if os.path.exists(SAVE_PATH):
        raise FileExistsError(SAVE_PATH)

    assert isinstance(RAND_SEED, int) and (RAND_SEED >= 0)

    np.random.seed(seed=RAND_SEED)
    torch.manual_seed(seed=RAND_SEED)

    sample_by_label: dict = oversampling(data_path=DATA_PATH,
                                         sampling_method=SAMPLING_METHOD,
                                         ratio_by_label=RATIO_BY_LABEL,
                                         smote_k_neighbors=SMOTE_K_NEIGHBORS,
                                         smote_svm_kernel=SMOTE_SVM_KERNEL,
                                         gan_size_latent=GAN_SIZE_LATENT,
                                         gan_num_hidden_layers=GAN_NUM_HIDDEN_LAYERS,
                                         rand_seed=RAND_SEED)

    save_dir: str = os.path.split(SAVE_PATH)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(SAVE_PATH, mode="wb") as fp:
        pickle.dump(sample_by_label, fp)

    print(">> Save the oversampled data: {0}".format((SAVE_PATH)))
