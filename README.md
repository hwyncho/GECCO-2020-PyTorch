# GECCO-2020-PyTorch

> A Genetic Algorithm to Optimize SMOTE and GAN Ratios in Class Imbalanced Datasets.

The experimental codes using PyTorch from the [paper](https://github.com/hwyncho/GECCO-2020-Paper) that was submitted to [GECCO 2020](https://gecco-2020.sigevo.org/index.html/HomePage). (https://doi.org/10.1145/3377929.3398153)

## Getting Started

### Environments

- Ubuntu 16.04 or 18.04
- Python 3.6 or 3.7

### Installation from `PyPi`

- PyTorch 1.4.0
- scikit-learn
- pandas
- DEAP
- imbalanced-learn

### Installation from `Docker`

- [`Dockerfile`](./Dockerfile)

## Codes

- [`classifier/`](./classifier)
  : A python module implementing NN-based classifier.

- [`ga/`](./ga)
  : A python module that implements the GA method to find the optimal oversampling ratio.

- [`gan/`](./gan)
  : A python module implementing GAN-based sampling method.

- [`oversample.py`](./smaple_dataset.py)
  : Executable script to oversample minority data using SMOTE, SVMSMOTE, GAN, etc.

- [`train.py`](./train.py)

- [`eval.py`](./eval.py)

- [`search_GA.py`](./search_GA.py)

- [`train_GA.py`](./train_GA.py)
