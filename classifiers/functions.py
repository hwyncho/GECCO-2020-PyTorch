"""Module containing the functions."""
import copy
import os
import time
from pprint import pprint
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import metrics as sklearn_metrics


def train(classifier: torch.nn.Module,
          x: torch.Tensor,
          y: torch.Tensor,
          batch_size: int = 16,
          num_epochs: int = 2,
          run_device: str = "cpu",
          learning_rate: float = 0.001,
          beta_1: float = 0.9,
          beta_2: float = 0.999,
          random_state: torch.Tensor = None,
          verbose: bool = False) -> Tuple[torch.nn.Module, torch.Tensor]:
    """
    Function to train classifiers and save the trained classifiers.

    Parameters
    ----------
    classifier: torch.nn.Module
    x: torch.Tensor
    y: torch.Tensor
    batch_size: int
    num_epochs: int
    run_device: str
    learning_rate: float
    beta_1: float
    beta_2: float
    random_state: torch.Tensor
    verbose: bool

    Returns
    -------
    Tuple[torch.nn.Module, torch.Tensor]

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(batch_size, int) and (batch_size > 0)
    assert isinstance(num_epochs, int) and (num_epochs > 0)
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])
    assert isinstance(learning_rate, float) and (learning_rate > 0.0)
    assert isinstance(beta_1, float) and (0.0 <= beta_1 < 1.0)
    assert isinstance(beta_2, float) and (0.0 <= beta_2 < 1.0)
    assert isinstance(verbose, bool)

    # Set the seed for generating random numbers.
    if random_state is not None:
        assert isinstance(random_state, torch.Tensor)
        torch.set_rng_state(random_state)

    # Set the classifier.
    classifier = copy.deepcopy(classifier.cpu())
    run_device = run_device.lower()
    if run_device == "cuda":
        assert torch.cuda.is_available()
        classifier = classifier.cuda()
        if torch.cuda.device_count() > 1:
            num_gpus: int = torch.cuda.device_count()
            classifier = torch.nn.DataParallel(classifier, device_ids=list(range(0, num_gpus)))

    # Set a criterion and optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=classifier.parameters(),
                                 lr=learning_rate,
                                 betas=(beta_1, beta_2))

    # Covert PyTorch's Tensor to TensorDataset.
    x, y = x.clone(), y.clone()
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             shuffle=True)

    # Train the classifiers.
    classifier.train()
    log: str = "[{0}/{1}] Loss: {2:.4f}, Time: {3:.2f}s"
    loss_list: list = list()
    for epoch in range(1, num_epochs + 1):
        start_time: float = time.time()
        for (_, batch) in enumerate(dataloader, 0):
            batch_x, batch_y = batch
            if run_device == "cuda":
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            output: torch.Tensor = classifier(batch_x)
            loss: torch.Tensor = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach().cpu().item())
        end_time: float = time.time()

        if verbose:
            print(log.format(epoch, num_epochs, np.mean(loss_list), end_time - start_time))

    if isinstance(classifier, torch.nn.DataParallel):
        classifier = classifier.module

    return classifier.cpu(), torch.get_rng_state()


def predict(classifier: torch.nn.Module,
            x: torch.Tensor,
            run_device: str = "cpu",
            random_state: torch.Tensor = None) -> np.ndarray:
    """
    Function to evaluate the trained classifiers.

    Parameters
    ----------
    classifier: torch.nn.Module
    x: torch.Tensor
    run_device: str
    random_state: torch.Tensor

    Returns
    -------
    np.ndarray

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])

    # Set the seed for generating random numbers.
    if random_state is not None:
        assert isinstance(random_state, torch.Tensor)
        torch.set_rng_state(random_state)

    # Set the classifiers.
    classifier = copy.deepcopy(classifier.cpu())
    run_device = run_device.lower()
    if run_device == "cuda":
        assert torch.cuda.is_available()
        classifier = classifier.cuda()
        if torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            classifier = torch.nn.DataParallel(classifier, device_ids=list(range(0, num_gpus)))

    # Covert PyTorch's Tensor to TensorDataset.
    x = x.clone()
    dataset = torch.utils.data.TensorDataset(x)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=x.size(0),
                                             num_workers=0,
                                             shuffle=False)

    batch_x: torch.Tensor = next(iter(dataloader))[0]
    if run_device == "cuda":
        batch_x = batch_x.cuda()

    classifier.eval()
    with torch.no_grad():
        output: torch.Tensor = classifier(batch_x)
        predict_y: torch.Tensor = output.detach().cpu().argmax(dim=1,
                                                               keepdim=True)

    return predict_y.numpy()


def evaluate(classifier: torch.nn.Module,
             x: torch.Tensor,
             y: torch.Tensor,
             metric: str = "f1_score",
             run_device: str = "cpu",
             random_state: torch.Tensor = None,
             verbose: bool = False) -> np.ndarray:
    """
    Function to evaluate the trained classifiers.

    Parameters
    ----------
    classifier: torch.nn.Module
    x: torch.Tensor
    y: torch.Tensor
    metric: str
    run_device: str
    random_state: torch.Tensor = None
    verbose: bool

    Returns
    -------
    numpy.ndarray

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(metric, str) and (metric.lower() in ["confusion_matrix", "f1_score"])
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])
    if random_state is not None:
        assert isinstance(random_state, torch.Tensor)
    assert isinstance(verbose, bool)

    run_device = run_device.lower()
    predict_y: np.ndarray = predict(classifier=classifier,
                                    x=x,
                                    run_device=run_device,
                                    random_state=random_state)

    num_labels: int = int(y.max().item() - y.min().item()) + 1
    metric = metric.lower()
    if metric == "confusion_matrix":
        confusion_matrix: np.ndarray = sklearn_metrics.confusion_matrix(y.numpy(),
                                                                        predict_y,
                                                                        labels=list(range(0, num_labels)))
        if verbose:
            df_cm: pd.DataFrame = pd.DataFrame(confusion_matrix)
            df_cm.columns = ["Predict_{0}".format(label) for label in range(0, num_labels)]
            df_cm.index = ["Real_{0}".format(label) for label in range(0, num_labels)]

            print(">> Confusion matrix :")
            pprint(df_cm)

        return confusion_matrix
    elif metric == "f1_score":
        f1_score: np.ndarray = sklearn_metrics.f1_score(y.numpy(),
                                                        predict_y,
                                                        labels=list(range(0, num_labels)),
                                                        average=None)

        if verbose:
            df_f1: pd.DataFrame = pd.DataFrame(f1_score[np.newaxis])
            df_f1.columns = ["Label_{0}".format(label) for label in range(0, num_labels)]
            df_f1.index = ["F1_score"]

            print(">> F1 score :")
            pprint(df_f1)

        return f1_score
    else:
        raise ValueError()


def save_model(classifier: torch.nn.Module, model_path: str, random_state: torch.Tensor) -> bool:
    """
    Function to save the parameters of the trained classifier.

    Parameters
    ----------
    classifier: torch.nn.Module
    model_path: str
    random_state: torch.Tensor

    Returns
    -------
    bool

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(model_path, str)
    assert os.path.splitext(model_path)[1].lower() in [".pt", ".pth"]

    model_path = os.path.abspath(model_path)
    model_pardir: str = os.path.split(model_path)[0]
    if not os.path.exists(model_pardir):
        os.makedirs(model_pardir)

    checkpoint: dict = {"classifier_state_dict": classifier.cpu().state_dict()}
    if random_state is not None:
        assert isinstance(random_state, torch.Tensor)
        checkpoint["random_state"] = random_state
    else:
        checkpoint["random_state"] = torch.get_rng_state()

    # Save the trained classifiers.
    torch.save(checkpoint, model_path)

    return True


def load_model(classifier: torch.nn.Module, model_path: str) -> Tuple[torch.nn.Module, torch.Tensor]:
    """
    Function to load the parameters from the trained classifier.

    Parameters
    ----------
    classifier: torch.nn.Module
    model_path: str

    Returns
    -------
    Tuple[torch.nn.Module, torch.Tensor]

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(model_path, str) and os.path.exists(model_path)
    assert os.path.splitext(model_path)[1].lower() in [".pt", ".pth"]

    checkpoint: dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Load the trained classifiers.
    classifier = copy.deepcopy(classifier.cpu())
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    if "random_state" in checkpoint:
        random_state = checkpoint["random_state"]
        return classifier, random_state
    else:
        return classifier, torch.get_rng_state()
