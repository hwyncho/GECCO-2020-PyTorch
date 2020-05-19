"""Module containing the mutate operators."""
import numpy as np


def mutate_swap(individual) -> tuple:
    """
    Swap two points.

    Parameters
    ----------

    Returns
    -------

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
