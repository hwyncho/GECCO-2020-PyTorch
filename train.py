#!/usr/bin/env python3
"""Train a classifiers."""
import argparse
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch

from classifiers import DNNClassifier
from classifiers import save_model
from classifiers import train


def load_dataset(data_path: str,
                 sampling_method: str = None,
                 ratio_by_label: dict = None,
                 samples_dir: str = None,
                 **kwargs) -> tuple:
    """
    Function to load the csv data file.

    Parameters
    ----------
    data_path: str
    sampling_method: str
    ratio_by_label: dict
    samples_dir: str

    Returns
    -------
    tuple

    """
    assert isinstance(data_path, str)
    if sampling_method:
        assert isinstance(sampling_method, str)
        assert sampling_method.lower() in ["smote", "smote_borderline", "smote_svm", "adasyn", "gan"]
        if ratio_by_label:
            assert isinstance(ratio_by_label, dict)
        assert isinstance(samples_dir, str)
        if not os.path.exists(samples_dir):
            raise FileNotFoundError(samples_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    assert (os.path.splitext(data_path)[1]).lower() == ".csv"

    smote_k_neighbors: int = 5
    if "smote_k_neighbors" in kwargs:
        assert isinstance(kwargs["smote_k_neighbors"], int) and (kwargs["smote_k_neighbors"] > 0)
        smote_k_neighbors = kwargs["smote_k_neighbors"]

    smote_borderline_kind: str = "borderline-1"
    if "smote_borderline_kind" in kwargs:
        assert isinstance(kwargs["smote_borderline_kind"], str)
        smote_borderline_kind = str(kwargs["smote_borderline_kind"]).lower()

    smote_svm_kernel: str = "rbf"
    if "smote_svm_kernel" in kwargs:
        assert isinstance(kwargs["smote_svm_kernel"], str)
        smote_svm_kernel = str(kwargs["smote_svm_kernel"]).lower()

    adasyn_n_neighbors: int = 5
    if "adasyn_n_neighbors" in kwargs:
        assert isinstance(kwargs["adasyn_n_neighbors"], int) and (kwargs["adasyn_n_neighbors"] > 0)
        adasyn_n_neighbors = kwargs["adasyn_n_neighbors"]

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

    x_np: np.ndarray = data_np[:, 0:-1]
    y_np: np.ndarray = data_np[:, -1]

    if sampling_method:
        assert ratio_by_label

        MAX_RATIO_BY_LABEL: str = "1=5.0"
        y_stats: dict = Counter(y_np)
        number_by_label: dict = dict()
        for (label, ratio) in ratio_by_label.items():
            label, ratio = int(label), float(ratio)
            assert ratio > 1.0
            if label in y_stats:
                number_by_label[label] = int(y_stats[label] * (ratio - 1.0))
            else:
                raise RuntimeError("{0} class is not in {1}.".format(label, data_path))

        sample_path: str = ""
        if sampling_method == "smote":
            sample_path = os.path.join(samples_dir,
                                       sampling_method,
                                       "k_neighbors={0}".format(smote_k_neighbors),
                                       "ratio_by_label={0}".format(str(MAX_RATIO_BY_LABEL)),
                                       "sample_by_label.pkl")
        elif sampling_method == "smote_borderline":
            sample_path = os.path.join(samples_dir,
                                       sampling_method,
                                       "k_neighbors={0}".format(smote_k_neighbors),
                                       "borderline_kind={0}".format(smote_borderline_kind),
                                       "ratio_by_label={0}".format(str(MAX_RATIO_BY_LABEL)),
                                       "sample_by_label.pkl")
        elif sampling_method == "smote_svm":
            sample_path = os.path.join(samples_dir,
                                       sampling_method,
                                       "k_neighbors={0}".format(smote_k_neighbors),
                                       "svm_kernel={0}".format(smote_svm_kernel),
                                       "ratio_by_label={0}".format(str(MAX_RATIO_BY_LABEL)),
                                       "sample_by_label.pkl")
        elif sampling_method == "adasyn":
            sample_path = os.path.join(samples_dir,
                                       sampling_method,
                                       "n_neighbors={0}".format(adasyn_n_neighbors),
                                       "ratio_by_label={0}".format(str(MAX_RATIO_BY_LABEL)),
                                       "sample_by_label.pkl")
        elif sampling_method == "gan":
            sample_path = os.path.join(samples_dir,
                                       sampling_method,
                                       "size_latent={0}".format(gan_size_latent),
                                       "num_hidden_layers={0}".format(gan_num_hidden_layers),
                                       "ratio_by_label={0}".format(str(MAX_RATIO_BY_LABEL)),
                                       "sample_by_label.pkl")

        if not os.path.exists(sample_path):
            raise FileNotFoundError(sample_path)
        with open(sample_path, mode="rb") as fp:
            sample = pickle.load(fp)

        for (label, number) in number_by_label.items():
            new_x: np.ndarray = sample[label][0][:number]
            new_y: np.ndarray = sample[label][1][:number]

            x_np = np.concatenate([x_np, new_x], axis=0)
            y_np = np.concatenate([y_np, new_y], axis=0)

    x_tensor: torch.Tensor = torch.as_tensor(x_np, dtype=torch.float)
    y_tensor: torch.Tensor = torch.as_tensor(y_np, dtype=torch.long)

    return x_tensor, y_tensor


def parse_args():
    class StoreDictKeyPair(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            my_dict: dict = dict()
            for kv in values.split(","):
                k, v = kv.split("=")
                my_dict[k] = v
            setattr(namespace, self.dest, my_dict)

    parser = argparse.ArgumentParser(description="Arguments for train a classifiers.")
    parser.add_argument("--train-data-path", type=str, required=True,
                        help="File path of a train data.",
                        dest="train_data_path")
    parser.add_argument("--sampling-method", type=str, default=None, required=False,
                        help="Name of method to oversample a dataset.",
                        dest="sampling_method")
    parser.add_argument("--ratio-by-label", action=StoreDictKeyPair, required=False,
                        metavar="KEY1=VAL1,KEY2=VAL2 ...",
                        help="Oversampling ratio by label.",
                        dest="ratio_by_label")
    parser.add_argument("--samples-dir", type=str, default=None, required=False,
                        help="Directory path of store samples.",
                        dest="samples_dir")
    parser.add_argument("--smote-k-neighbors", type=int, default=5, required=False,
                        help="Parameter k_neighbors of SMOTE.",
                        dest="smote_k_neighbors")
    parser.add_argument("--smote-borderline-kind", type=str, default="borderline-1", required=False,
                        help="Parameter borderline_kind of BorderlineSMOTE.",
                        dest="smote_borderline_kind")
    parser.add_argument("--smote-svm-kernel", type=str, default="linear", required=False,
                        help="Parameter svm_kernel of SVMSMOTE.",
                        dest="smote_svm_kernel")
    parser.add_argument("--adasyn-n-neighbors", type=int, default=5, required=False,
                        help="Parameter n_neighbors of ADASYN.",
                        dest="adasyn_n_neighbors")
    parser.add_argument("--gan-size-latent", type=int, default=100, required=False,
                        help="Parameter size_latent of GAN.",
                        dest="gan_size_latent")
    parser.add_argument("--gan-num-hidden-layers", type=int, default=2, required=False,
                        help="Parameter num_hidden_layers of GAN.",
                        dest="gan_num_hidden_layers")
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
    SAMPLING_METHOD: str = args.sampling_method
    RATIO_BY_LABEL: dict = args.ratio_by_label
    SAMPLES_DIR: str = args.samples_dir
    SMOTE_K_NEIGHBORS: int = args.smote_k_neighbors
    SMOTE_BORDERLINE_KIND: str = args.smote_borderline_kind
    SMOTE_SVM_KERNEL: str = args.smote_svm_kernel
    ADASYN_N_NEIGHBORS: int = args.adasyn_n_neighbors
    GAN_SIZE_LATENT: int = args.gan_size_latent
    GAN_NUM_HIDDEN_LAYERS: int = args.gan_num_hidden_layers
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
    torch_random_generator = torch.manual_seed(seed=RAND_SEED)

    numpy_random_state = np.random.get_state()
    torch_random_state = torch_random_generator.get_state()

    x, y = load_dataset(data_path=TRAIN_DATA_PATH,
                        sampling_method=SAMPLING_METHOD,
                        ratio_by_label=RATIO_BY_LABEL,
                        samples_dir=SAMPLES_DIR,
                        smote_k_neighbors=SMOTE_K_NEIGHBORS,
                        smote_borderline_kind=SMOTE_BORDERLINE_KIND,
                        smote_svm_kernel=SMOTE_SVM_KERNEL,
                        adasyn_n_neighbors=ADASYN_N_NEIGHBORS,
                        gan_size_latent=GAN_SIZE_LATENT,
                        gan_num_hidden_layers=GAN_NUM_HIDDEN_LAYERS)
    size_features: int = x.size(1)
    size_labels: int = int(y.max().item() - y.min().item()) + 1

    # Train a classifiers and save the classifiers.
    classifier: torch.nn.Module = DNNClassifier(size_features=size_features,
                                                num_hidden_layers=NUM_HIDDEN_LAYERS,
                                                size_labels=size_labels)
    trained_classifier, trained_random_state = train(classifier=classifier,
                                                     x=x,
                                                     y=y,
                                                     batch_size=BATCH_SIZE,
                                                     num_epochs=NUM_EPOCHS,
                                                     run_device=RUN_DEVICE,
                                                     learning_rate=LEARNING_RATE,
                                                     beta_1=BETA_1,
                                                     beta_2=BETA_2,
                                                     random_state=torch_random_state,
                                                     verbose=VERBOSE)
    save_model(classifier=trained_classifier, model_path=MODEL_PATH, random_state=trained_random_state)

    print(">> Save the trained classifier: {0}".format((MODEL_PATH)))
