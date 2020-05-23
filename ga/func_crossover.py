"""Module containing the crossover operators."""
import numpy as np


def crossover_onepoint(individual_1, individual_2) -> tuple:
    """
    One-point Crossover.

    Parameters
    ----------
    individual_1
    individual_2

    Returns
    -------
    tuple

    """

    def crossover_row(row_1, row_2, point: int) -> tuple:
        assert isinstance(point, int) and (point < len(row_1))

        a, b = row_1.copy(), row_2.copy()
        a[point:], b[point:] = b[point:].copy(), a[point:].copy()

        return a, b

    len_y, len_x = len(individual_1), len(individual_1[0])
    point_y, point_x = 0, 0
    if len_y >= 2:
        point_y = np.random.randint(low=1, high=len_y)
    if len_x >= 2:
        point_x = np.random.randint(low=1, high=len_x)

    if (point_y == 0) and (point_x == 0):
        return individual_2, individual_1
    else:
        if point_y == 0:
            individual_1[0], individual_2[0] = crossover_row(row_1=individual_1[0],
                                                             row_2=individual_2[0],
                                                             point=point_x)
        elif point_x == 0:
            row_1_t, row_2_t = individual_1.transpose(), individual_2.transpose()
            row_1, row_2 = crossover_row(row_1=row_1_t[0], row_2=row_2_t[0], point=point_y)
            individual_1 = row_1.reshape((1,) + row_1.shape).transpose()
            individual_2 = row_2.reshape((1,) + row_2.shape).transpose()
        else:
            for i in range(len_y):
                if i < point_y:
                    individual_1[i], individual_2[i] = crossover_row(row_1=individual_1[i],
                                                                     row_2=individual_2[i],
                                                                     point=point_x)
                else:
                    individual_1[i], individual_2[i] = crossover_row(row_1=individual_2[i],
                                                                     row_2=individual_1[i],
                                                                     point=point_x)

        return individual_1, individual_2


def crossover_twopoint(individual_1, individual_2) -> tuple:
    """
    Two-point Crossover.

    Parameters
    ----------
    individual_1
    individual_2

    Returns
    -------
    tuple

    """

    def crossover_row(row_1, row_2, point_1: int, point_2: int) -> tuple:
        assert isinstance(point_1, int) and (point_1 < len(row_1))
        assert isinstance(point_2, int) and (point_2 < len(row_1)) and (point_1 < point_2)

        a, b = row_1.copy(), row_2.copy()
        a[point_1:point_2], b[point_1:point_2] = b[point_1:point_2].copy(), a[point_1:point_2].copy()

        return a, b

    len_y, len_x = len(individual_1), len(individual_1[0])
    point_y_1, point_y_2 = 0, 1
    point_x_1, point_x_2 = 0, 1
    if len_y >= 3:
        point_y_1, point_y_2 = np.random.randint(low=1, high=len_y), np.random.randint(low=1, high=len_y)
        if point_y_1 > point_y_2:
            point_y_1, point_y_2 = point_y_2, point_y_1
        elif point_y_1 == point_y_2:
            if (point_y_1 + 1) == len_y:
                point_y_1 = point_y_2 - 1
            else:
                point_y_2 = point_y_1 + 1
    if len_x >= 3:
        point_x_1, point_x_2 = np.random.randint(low=1, high=len_x), np.random.randint(low=1, high=len_x)
        if point_x_1 > point_x_2:
            point_x_1, point_x_2 = point_x_2, point_x_1
        elif point_x_1 == point_x_2:
            if (point_x_1 + 1) == len_x:
                point_x_1 = point_x_2 - 1
            else:
                point_x_2 = point_x_1 + 1

    if (point_y_1 == 0) and (point_x_1 == 0):
        return individual_2, individual_1
    else:
        if point_y_1 == 0:
            individual_1[0], individual_2[0] = crossover_row(row_1=individual_1[0],
                                                             row_2=individual_2[0],
                                                             point_1=point_x_1,
                                                             point_2=point_x_2)
        elif point_x_1 == 0:
            row_1_t, row_2_t = individual_1.transpose(), individual_2.transpose()
            row_1, row_2 = crossover_row(row_1=row_1_t[0],
                                         row_2=row_2_t[0],
                                         point_1=point_y_1,
                                         point_2=point_y_2)
            individual_1 = row_1.reshape((1,) + row_1.shape).transpose()
            individual_2 = row_2.reshape((1,) + row_2.shape).transpose()
        else:
            for i in range(len_y):
                if point_y_1 <= i <= point_y_2:
                    individual_1[i], individual_2[i] = crossover_row(row_1=individual_2[i],
                                                                     row_2=individual_1[i],
                                                                     point_1=point_x_1,
                                                                     point_2=point_x_2)
                else:
                    individual_1[i], individual_2[i] = crossover_row(row_1=individual_1[i],
                                                                     row_2=individual_2[i],
                                                                     point_1=point_x_1,
                                                                     point_2=point_x_2)

        return individual_1, individual_2


def crossover_uniform(individual_1, individual_2) -> tuple:
    """
    Uniform Crossover.

    Parameters
    ----------
    individual_1
    individual_2

    Returns
    -------
    tuple

    """

    def crossover_row(row_1, row_2) -> tuple:
        a, b = row_1.copy(), row_2.copy()
        arr: np.ndarray = np.random.random(size=(len(row_1),))
        points: np.ndarray = np.where(arr >= 0.5)[0]
        a[points], b[points] = b[points].copy(), a[points].copy()

        return a, b

    len_y, len_x = len(individual_1), len(individual_1[0])
    if (len_y == 1) and (len_x == 1):
        return individual_2, individual_1
    else:
        if len_y == 1:
            individual_1[0], individual_2[0] = crossover_row(row_1=individual_1[0], row_2=individual_2[0])
        elif len_x == 1:
            row_1_t, row_2_t = individual_1.transpose(), individual_2.transpose()
            row_1, row_2 = crossover_row(row_1=row_1_t[0], row_2=row_2_t[0])
            individual_1 = row_1.reshape((1,) + row_1.shape).transpose()
            individual_2 = row_2.reshape((1,) + row_2.shape).transpose()
        else:
            for i in range(len_y):
                individual_1[i], individual_2[i] = crossover_row(row_1=individual_1[i], row_2=individual_2[i])

        return individual_1, individual_2


def crossover_pmx(individual_1, individual_2) -> tuple:
    """
    PMX Crossover.

    Parameters
    ----------
    individual_1
    individual_2

    Returns
    -------
    tuple

    """

    def crossover_row(row_1, row_2) -> tuple:
        a, b = row_1.copy(), row_2.copy()
        arr_1: np.ndarray = np.zeros(shape=(len(a),), dtype=np.int)
        arr_2: np.ndarray = np.zeros(shape=(len(a),), dtype=np.int)

        for i in range(len(arr_1)):
            arr_1[a[i]] = i
            arr_2[b[i]] = i

        point_1: int = np.random.randint(low=0, high=len(a) - 1)
        point_2: int = np.random.randint(low=1, high=len(a))
        if point_1 == point_2:
            point_2 = point_1 + 1
        elif point_1 > point_2:
            point_1, point_2 = point_2, point_1

        for i in range(point_1, point_2):
            temp_1, temp_2 = a[i], b[i]

            a[i], a[arr_1[temp_2]] = temp_2, temp_1
            b[i], b[arr_2[temp_1]] = temp_1, temp_2

            arr_1[temp_1], arr_1[temp_2] = arr_1[temp_2], arr_1[temp_1]
            arr_2[temp_1], arr_2[temp_2] = arr_2[temp_2], arr_2[temp_1]

        return a, b

    len_y, len_x = len(individual_1), len(individual_1[0])
    if (len_y == 1) and (len_x == 1):
        return individual_2, individual_1
    else:
        if len_y == 1:
            individual_1[0], individual_2[0] = crossover_row(row_1=individual_1[0], row_2=individual_2[0])
        elif len_x == 1:
            row_1_t, row_2_t = individual_1.transpose(), individual_2.transpose()
            row_1, row_2 = crossover_row(row_1=row_1_t[0], row_2=row_2_t[0])
            individual_1 = row_1.reshape((1,) + row_1.shape).transpose()
            individual_2 = row_2.reshape((1,) + row_2.shape).transpose()
        else:
            for i in range(len_y):
                individual_1[i], individual_2[i] = crossover_row(row_1=individual_1[i], row_2=individual_2[i])

        return individual_1, individual_2
