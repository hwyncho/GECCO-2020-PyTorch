"""Module containing the functions."""
import functools
import random

import numpy as np
import torch
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from .func_crossover import crossover_onepoint
from .func_fitness import calculate_f1_score
from .func_mutate import mutate_swap


def generate_individual(ratio_min: float = 1.0,
                        ratio_max: float = 1.1,
                        num_sampling_methods: int = 1,
                        num_sampling_labels: int = 1) -> np.ndarray:
    assert isinstance(ratio_min, float) and (ratio_min >= 1.0)
    assert isinstance(ratio_max, float) and (ratio_max > ratio_min)
    assert isinstance(num_sampling_methods, int) and (num_sampling_methods >= 1)
    assert isinstance(num_sampling_labels, int) and (num_sampling_labels >= 1)

    ratio_sum: float = np.random.uniform(low=ratio_min, high=ratio_max, size=(num_sampling_labels, 1))
    individual: np.ndarray = np.random.dirichlet(alpha=np.ones(num_sampling_methods),
                                                 size=num_sampling_labels) * ratio_sum

    return np.asarray(individual, dtype=np.float32)


def run(x: torch.Tensor,
        y: torch.Tensor,
        list_sample_by_label: list,
        ratio_min: float = 1.0,
        ratio_max: float = 1.1,
        population_size: int = 4,
        crossover_size: int = 2,
        mutation_rate: float = 0.01,
        num_generations: int = 1,
        rand_seed: int = 0,
        verbose: bool = False,
        **kwargs) -> tuple:
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(list_sample_by_label, list)
    assert isinstance(ratio_min, float) and (ratio_min >= 1.0)
    assert isinstance(ratio_max, float) and (ratio_max > ratio_min)
    assert isinstance(population_size, int) and (population_size > 0)
    assert isinstance(crossover_size, int) and (1 < crossover_size <= population_size)
    assert isinstance(mutation_rate, float) and (0.0 <= mutation_rate <= 1.0)
    assert isinstance(num_generations, int) and num_generations >= 1
    assert isinstance(rand_seed, int) and rand_seed > 0
    assert isinstance(verbose, bool)

    num_hidden_layers: int = 2
    if "classifier_num_hidden_layers" in kwargs:
        assert isinstance(kwargs["classifier_num_hidden_layers"], int) and (kwargs["classifier_num_hidden_layers"] > 0)
        num_hidden_layers = kwargs["classifier_num_hidden_layers"]

    # Parameters for Adam optimizer.
    learning_rate: float = 0.001
    if "classifier_learning_rate" in kwargs:
        assert isinstance(kwargs["classifier_learning_rate"], float) and (kwargs["classifier_learning_rate"] > 0.0)
        learning_rate = kwargs["classifier_learning_rate"]
    beta_1: float = 0.9
    if "classifier_beta_1" in kwargs:
        assert isinstance(kwargs["classifier_beta_1"], float) and (0.0 <= kwargs["classifier_beta_1"] < 1.0)
        beta_1 = kwargs["classifier_beta_1"]
    beta_2: float = 0.999
    if "classifier_beta_2" in kwargs:
        assert isinstance(kwargs["classifier_beta_2"], float) and (0.0 <= kwargs["classifier_beta_2"] < 1.0)
        beta_2 = kwargs["classifier_beta_2"]

    # Parameters for training.
    batch_size: int = 1024
    if "classifier_batch_size" in kwargs:
        assert isinstance(kwargs["classifier_batch_size"], int) and (kwargs["classifier_batch_size"] > 0)
        batch_size = kwargs["classifier_batch_size"]
    num_epochs: int = 10
    if "classifier_num_epochs" in kwargs:
        assert isinstance(kwargs["classifier_num_epochs"], int) and (kwargs["classifier_num_epochs"] > 0)
        num_epochs = kwargs["classifier_num_epochs"]

    # Check the running device for PyTorch.
    run_device: str = "cpu"
    if "classifier_run_device" in kwargs:
        assert isinstance(kwargs["classifier_run_device"], str)
        assert str(kwargs["classifier_run_device"]).lower() in ["cpu", "cuda"]
        run_device = str(kwargs["classifier_run_device"]).lower()

    random.seed(a=rand_seed)
    np.random.seed(seed=rand_seed)

    size_labels: int = int(y.max().item() - y.min().item()) + 1

    assert isinstance(list_sample_by_label[0], dict)
    num_sampling_methods: int = len(list_sample_by_label)
    num_sampling_labels: int = len(list_sample_by_label[0].keys())

    cxpb: float = float(crossover_size) / float(population_size)

    func_generate_individual = functools.partial(generate_individual,
                                                 ratio_min=ratio_min,
                                                 ratio_max=ratio_max,
                                                 num_sampling_methods=num_sampling_methods,
                                                 num_sampling_labels=num_sampling_labels)

    creator.create(name="FitnessMax", base=base.Fitness, weights=tuple([1.0 for _ in range(size_labels)]))
    creator.create(name="Individual", base=np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register(alias="individual", function=tools.initIterate,
                     container=creator.Individual, generator=func_generate_individual)

    toolbox.register(alias="population", function=tools.initRepeat,
                     container=list, func=toolbox.individual)

    toolbox.register(alias="evaluate", function=calculate_f1_score,
                     x=x,
                     y=y,
                     num_hidden_layers=num_hidden_layers,
                     list_sample_by_label=list_sample_by_label,
                     rand_seed=rand_seed,
                     learning_rate=learning_rate,
                     beta_1=beta_1,
                     beta_2=beta_2,
                     batch_size=batch_size,
                     num_epochs=num_epochs,
                     run_device=run_device.lower())

    # toolbox.register(alias="select", function=tools.selTournament, tournsize=4)
    toolbox.register(alias="select", function=tools.selRoulette)

    toolbox.register(alias="mate", function=crossover_onepoint)

    toolbox.register(alias="mutate", function=mutate_swap)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(4, similar=np.array_equal)

    if verbose:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)

        algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutation_rate, ngen=num_generations,
                            stats=stats, halloffame=hof)

        return hof, (population, stats)
    else:
        algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutation_rate, ngen=num_generations,
                            stats=None, halloffame=hof)

        return hof, (population,)
