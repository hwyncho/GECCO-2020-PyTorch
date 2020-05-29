"""Module containing the fitness function."""
from collections import Counter
from typing import Tuple

import numpy as np
import torch

from classifiers import DNNClassifier
from classifiers import evaluate as classifier_evaluate
from classifiers import train as classifier_train


def calculate_f1_score(individual: np.ndarray,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       list_sample_by_label: list,
                       random_state: torch.Tensor = None,
                       **kwargs) -> Tuple:
    """
    Function to calculate fitness.

    Parameters
    ----------
    individual: np.ndarray
    x: torch.Tensor
    y: torch.Tensor
    list_sample_by_label: list
    random_state: torch.Tensor

    Returns
    -------
    Tuple

    """
    assert isinstance(individual, np.ndarray)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(list_sample_by_label, list)
    if random_state is not None:
        assert isinstance(random_state, torch.Tensor)

    classifier_num_hidden_layers: int = 1
    if "classifier_num_hidden_layers" in kwargs:
        assert isinstance(kwargs["classifier_num_hidden_layers"], int) and (kwargs["classifier_num_hidden_layers"] > 0)
        classifier_num_hidden_layers = kwargs["classifier_num_hidden_layers"]

    # Parameters for training.
    classifier_batch_size: int = 16
    if "classifier_batch_size" in kwargs:
        assert isinstance(kwargs["classifier_batch_size"], int) and (kwargs["classifier_batch_size"] > 0)
        classifier_batch_size = kwargs["classifier_batch_size"]
    classifier_num_epochs: int = 2
    if "classifier_num_epochs" in kwargs:
        assert isinstance(kwargs["classifier_num_epochs"], int) and (kwargs["classifier_num_epochs"] > 0)
        classifier_num_epochs = kwargs["classifier_num_epochs"]

    # Check the running device for PyTorch.
    classifier_run_device: str = "cpu"
    if "classifier_run_device" in kwargs:
        assert isinstance(kwargs["classifier_run_device"], str)
        assert str(kwargs["classifier_run_device"]).lower() in ["cpu", "cuda"]
        classifier_run_device = str(kwargs["classifier_run_device"]).lower()

    # Parameters for Adam optimizer.
    classifier_learning_rate: float = 0.001
    if "classifier_learning_rate" in kwargs:
        assert isinstance(kwargs["classifier_learning_rate"], float) and (kwargs["classifier_learning_rate"] > 0.0)
        classifier_learning_rate = kwargs["classifier_learning_rate"]
    classifier_beta_1: float = 0.9
    if "classifier_beta_1" in kwargs:
        assert isinstance(kwargs["classifier_beta_1"], float) and (0.0 <= kwargs["classifier_beta_1"] < 1.0)
        classifier_beta_1 = kwargs["classifier_beta_1"]
    classifier_beta_2: float = 0.999
    if "classifier_beta_2" in kwargs:
        assert isinstance(kwargs["classifier_beta_2"], float) and (0.0 <= kwargs["classifier_beta_2"] < 1.0)
        classifier_beta_2 = kwargs["classifier_beta_2"]

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
                                                num_hidden_layers=classifier_num_hidden_layers,
                                                size_labels=size_labels)
    trained_classifier, trained_random_state = classifier_train(classifier=classifier,
                                                                x=train_x,
                                                                y=train_y,
                                                                batch_size=classifier_batch_size,
                                                                num_epochs=classifier_num_epochs,
                                                                run_device=classifier_run_device,
                                                                learning_rate=classifier_learning_rate,
                                                                beta_1=classifier_beta_1,
                                                                beta_2=classifier_beta_2,
                                                                random_state=random_state,
                                                                verbose=False)
    f1_score: np.ndarray = classifier_evaluate(classifier=trained_classifier,
                                               x=x,
                                               y=y,
                                               metric="f1_score",
                                               run_device=classifier_run_device,
                                               random_state=trained_random_state,
                                               verbose=False)

    return tuple(f1_score.tolist(), )
