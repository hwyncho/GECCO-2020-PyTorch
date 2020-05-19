"""Module containing the fitness function."""
from collections import Counter

import numpy as np
import torch

from classifiers import DNNClassifier
from classifiers import evaluate as classifier_evaluate
from classifiers import train as classifier_train


def calculate_f1_score(individual: np.ndarray,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       list_sample_by_label: list,
                       rand_seed: int = 0,
                       **kwargs) -> tuple:
    """
    Function to calculate fitness.

    Parameters
    ----------
    individual: np.ndarray
    x: torch.Tensor
    y: torch.Tensor
    list_sample_by_label: list
    rand_seed: int

    Returns
    -------
    tuple

    """
    assert isinstance(individual, np.ndarray)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(list_sample_by_label, list)
    assert isinstance(rand_seed, int) and (rand_seed >= 0)

    num_hidden_layers: int = 1
    if "num_hidden_layers" in kwargs:
        assert isinstance(kwargs["num_hidden_layers"], int) and (kwargs["num_hidden_layers"] > 0)
        num_hidden_layers = kwargs["num_hidden_layers"]

    # Parameters for Adam optimizer.
    learning_rate: float = 0.001
    if "learning_rate" in kwargs:
        assert isinstance(kwargs["learning_rate"], float) and (kwargs["learning_rate"] > 0.0)
        learning_rate = kwargs["learning_rate"]
    beta_1: float = 0.9
    if "beta_1" in kwargs:
        assert isinstance(kwargs["beta_1"], float) and (0.0 <= kwargs["beta_1"] < 1.0)
        beta_1 = kwargs["beta_1"]
    beta_2: float = 0.999
    if "beta_2" in kwargs:
        assert isinstance(kwargs["beta_2"], float) and (0.0 <= kwargs["beta_2"] < 1.0)
        beta_2 = kwargs["beta_2"]

    # Parameters for training.
    batch_size: int = 1024
    if "batch_size" in kwargs:
        assert isinstance(kwargs["batch_size"], int) and (kwargs["batch_size"] > 0)
        batch_size = kwargs["batch_size"]
    num_epochs: int = 10
    if "num_epochs" in kwargs:
        assert isinstance(kwargs["num_epochs"], int) and (kwargs["num_epochs"] > 0)
        num_epochs = kwargs["num_epochs"]

    # Check the running device for PyTorch.
    run_device: str = "cpu"
    if "run_device" in kwargs:
        assert isinstance(kwargs["run_device"], str) and (str(kwargs["run_device"]).lower() in ["cpu", "cuda"])
        run_device = str(kwargs["run_device"]).lower()

    size_features: int = x.size(1)
    size_labels: int = int(y.max().item() - y.min().item()) + 1

    y_stats: dict = Counter(y.numpy())
    list_label: list = list(list_sample_by_label[0].keys())

    train_x: torch.Tensor = x.clone()
    train_y: torch.Tensor = y.clone()
    for (i, ratio_by_label) in enumerate(individual):
        label: int = list_label[i]
        for (j, ratio_by_method) in enumerate(ratio_by_label):
            method: int = j
            if ratio_by_method > 0.0:
                number: int = int(y_stats[label] * ratio_by_method)
                new_x: torch.Tensor = torch.as_tensor(list_sample_by_label[method][label][0][:number],
                                                      dtype=torch.float)
                new_y: torch.Tensor = torch.as_tensor(list_sample_by_label[method][label][1][:number],
                                                      dtype=torch.long)

                train_x = torch.cat([train_x, new_x], dim=0)
                train_y = torch.cat([train_y, new_y], dim=0)

    classifier: torch.nn.Module = DNNClassifier(size_features=size_features,
                                                num_hidden_layers=num_hidden_layers,
                                                size_labels=size_labels)
    trained_classifier: torch.nn.Module = classifier_train(classifier=classifier,
                                                           x=train_x,
                                                           y=train_y,
                                                           batch_size=batch_size,
                                                           num_epochs=num_epochs,
                                                           run_device=run_device,
                                                           learning_rate=learning_rate,
                                                           beta_1=beta_1,
                                                           beta_2=beta_2,
                                                           rand_seed=rand_seed,
                                                           verbose=False)
    f1_score: np.ndarray = classifier_evaluate(classifier=trained_classifier,
                                               x=x,
                                               y=y,
                                               metric="f1_score",
                                               run_device=run_device,
                                               rand_seed=rand_seed,
                                               verbose=False)

    return tuple(f1_score.tolist())
