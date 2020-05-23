"""Module containing the replacement operators."""
import numpy as np


def replacement_parents(population, parents, offspring):
    """
    Replace parents with offspring.

    Parameters
    ----------
    population
    parents
    offspring

    Returns
    -------

    """
    for i in range(len(offspring)):
        p = parents[i]
        idx = np.where((population == p).all(axis=1))[0]
        if len(idx) > 0:
            population[i] = offspring[i]

    return population


def replacement_parents_worse(population, parents, offspring):
    """
    Replace worse parents with offspring.

    Parameters
    ----------
    population
    parents
    offspring

    Returns
    -------

    """
    for i in range(len(offspring)):
        p = parents[i]
        if p.fitness.values < offspring[i].fitness.values:
            idx = np.where((population == p).all(axis=1))[0]
            if len(idx) > 0:
                population[i] = offspring[i]

    return population


def replacement_parents_better(population, parents, offspring):
    """
    Replace worse parents with offspring.

    Parameters
    ----------
    population
    parents
    offspring

    Returns
    -------

    """
    for i in range(len(offspring)):
        p = parents[i]
        if p.fitness.values > offspring[i].fitness.values:
            idx = np.where((population == p).all(axis=1))[0]
            if len(idx) > 0:
                population[i] = offspring[i]

    return population


def replacement_worst(population, parents, offspring):
    """
    Replace worst individuals with offspring.

    Parameters
    ----------
    population
    parents
    offspring

    Returns
    -------

    """
    population[-len(offspring):] = offspring[:]

    return population


def replacement_best(population, parents, offspring):
    """
    Replace worst individuals with offspring.

    Parameters
    ----------
    population
    parents
    offspring

    Returns
    -------

    """
    population[:len(offspring)] = offspring[:]

    return population
