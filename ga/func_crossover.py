"""Module containing the crossover operators."""
import numpy as np


def crossover_onepoint(individual_1, individual_2) -> tuple:
    """
    One-point Crossvoer.

    Parameters
    ----------
    individual_1,
    individual_2

    Returns
    -------
    tuple

    """

    def crossover_1d(row_1, row_2, point: int) -> tuple:
        a, b = row_1.copy(), row_2.copy()
        a[point:], b[point:] = b[point:].copy(), a[point:].copy()
        return a, b

    len_y: int = len(individual_1)
    point_y: int = 0
    if len_y >= 2:
        point_y = np.random.randint(low=1, high=len_y)

    len_x: int = len(individual_1[0])
    point_x: int = 0
    if len_x >= 2:
        point_x = np.random.randint(low=1, high=len_x)

    if (point_y == 0) and (point_x == 0):
        return individual_2, individual_1
    else:
        if point_y == 0:
            individual_1[0], individual_2[0] = crossover_1d(row_1=individual_1[0],
                                                            row_2=individual_2[0],
                                                            point=point_x)
        elif point_x == 0:
            row_1_t, row_2_t = np.transpose(individual_1), np.transpose(individual_2)
            row_1, row_2 = crossover_1d(row_1=row_1_t[0], row_2=row_2_t[0], point=point_y)
            individual_1, individual_2 = np.transpose([row_1]), np.transpose([row_2])
        else:
            for i in range(len_y):
                if i <= point_y:
                    individual_1[i], individual_2[i] = crossover_1d(row_1=individual_1[i],
                                                                    row_2=individual_2[i],
                                                                    point=point_x)
                else:
                    individual_1[i], individual_2[i] = crossover_1d(row_1=individual_2[i],
                                                                    row_2=individual_1[i],
                                                                    point=point_x)

        return individual_1, individual_2
