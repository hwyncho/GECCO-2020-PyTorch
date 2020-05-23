"""Module containing the mutation operators."""
import numpy as np


def mutation_swap(individual, **kwargs) -> tuple:
    """
    Swap two points.

    Parameters
    ----------
    individual

    Returns
    -------
    tuple

    """
    len_y: int = len(individual)
    point_y: int = 0
    if len_y >= 2:
        point_y = np.random.randint(low=0, high=len_y)

    len_x: int = len(individual[0])
    point_x_1: int = 0
    point_x_2: int = 0
    if len_x >= 2:
        point_x_1 = np.random.randint(low=0, high=len_x)
        point_x_2 = np.random.randint(low=0, high=len_x)

    a = individual[point_y][point_x_1]
    b = individual[point_y][point_x_2]

    individual[point_y][point_x_1] = b
    individual[point_y][point_x_2] = a

    return individual,


def mutation_range(individual, **kwargs) -> tuple:
    """

    Parameters
    ----------
    individual

    Returns
    -------
    tuple

    """
    ratio_min: float = 1.0
    if "ratio_min" in kwargs:
        ratio_min = kwargs["ratio_min"]

    ratio_max: float = 1.1
    if "ratio_max" in kwargs:
        ratio_max = kwargs["ratio_max"]

    len_y: int = len(individual)
    point_y: int = 0
    if len_y >= 2:
        point_y = np.random.randint(low=0, high=len_y)

    len_x: int = len(individual[0])
    point_x: int = 0
    if len_x >= 2:
        point_x = np.random.randint(low=0, high=len_x)

    sum_ = individual[point_y].sum()
    ratio_sum: float = np.random.uniform(low=ratio_min, high=ratio_max)
    diff = sum_ - ratio_sum
    individual[point_y][point_x] = individual[point_y][point_x] - diff

    return individual,


def mutation_shuffle(individual, **kwargs) -> tuple:
    """
    Shuffle.

    Parameters
    ----------
    individual

    Returns
    -------
    tuple

    """
    len_y: int = len(individual)
    point_y: int = 0
    if len_y >= 2:
        point_y = np.random.randint(low=0, high=len_y)

    points: np.ndarray = np.arange(start=0, stop=len(individual))
    np.random.shuffle(points)
    individual[point_y] = individual[point_y].take(points)

    return individual,
