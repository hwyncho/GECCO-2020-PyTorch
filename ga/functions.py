"""Module containing the functions."""
import functools
import random

import numpy as np
import torch
from deap import base
from deap import creator
from deap import tools

from .func_crossover import crossover_onepoint
from .func_crossover import crossover_pmx
from .func_crossover import crossover_twopoint
from .func_crossover import crossover_uniform
from .func_fitness import calculate_f1_score
from .func_mutation import mutation_range
from .func_mutation import mutation_shuffle
from .func_mutation import mutation_swap
from .func_replacement import replacement_best
from .func_replacement import replacement_parents
from .func_replacement import replacement_parents_better
from .func_replacement import replacement_parents_worse
from .func_replacement import replacement_worst


def generate_individual(ratio_min: float = 1.0,
                        ratio_max: float = 1.1,
                        num_sampling_methods: int = 1,
                        num_sampling_labels: int = 1) -> np.ndarray:
    assert isinstance(ratio_min, float) and (ratio_min >= 1.0)
    assert isinstance(ratio_max, float) and (ratio_max > ratio_min)
    assert isinstance(num_sampling_methods, int) and (num_sampling_methods >= 1)
    assert isinstance(num_sampling_labels, int) and (num_sampling_labels >= 1)

    ratio_sum: float = np.random.uniform(low=ratio_min, high=ratio_max, size=(num_sampling_labels, 1))
    alpha: np.ndarray = np.random.randint(low=1, high=11, size=num_sampling_methods)
    individual: np.ndarray = np.random.dirichlet(alpha=alpha,
                                                 size=num_sampling_labels) * ratio_sum

    return np.asarray(individual, dtype=np.float32)


def run(x: torch.Tensor,
        y: torch.Tensor,
        list_sample_by_label: list,
        ratio_min: float = 1.0,
        ratio_max: float = 1.1,
        population_size: int = 4,
        selection_method: str = "roulette",
        crossover_method: str = "onepoint",
        crossover_size: int = 2,
        mutation_method: str = "swap",
        mutation_rate: float = 0.01,
        replacement_method: str = "parents",
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
    assert isinstance(selection_method, str)
    assert selection_method.lower() in ["roulette", "tournament", "worst", "best", "random"]
    assert isinstance(crossover_method, str)
    assert crossover_method.lower() in ["onepoint", "twopoint", "uniform", "pmx"]
    assert isinstance(crossover_size, int) and (1 < crossover_size <= population_size)
    assert isinstance(mutation_method, str)
    assert mutation_method.lower() in ["swap", "range", "shuffle"]
    assert isinstance(mutation_rate, float) and (0.0 <= mutation_rate <= 1.0)
    assert isinstance(replacement_method, str)
    assert replacement_method.lower() in ["parents", "parents_worse", "parents_better", "worst", "best"]
    assert isinstance(num_generations, int) and num_generations >= 1
    assert isinstance(rand_seed, int) and rand_seed > 0
    assert isinstance(verbose, bool)

    # Parameters for NN-based classifier.
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

    random.seed(a=rand_seed)
    np.random.seed(seed=rand_seed)

    size_labels: int = int(y.max().item() - y.min().item()) + 1

    assert isinstance(list_sample_by_label[0], dict)
    num_sampling_methods: int = len(list_sample_by_label)
    num_sampling_labels: int = len(list_sample_by_label[0].keys())

    # Set a genetic algorithm.
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
                     list_sample_by_label=list_sample_by_label,
                     rand_seed=rand_seed,
                     classifier_num_hidden_layers=classifier_num_hidden_layers,
                     classifier_batch_size=classifier_batch_size,
                     classifier_num_epochs=classifier_num_epochs,
                     classifier_run_device=classifier_run_device.lower(),
                     classifier_learning_rate=classifier_learning_rate,
                     classifier_beta_1=classifier_beta_1,
                     classifier_beta_2=classifier_beta_2)

    # Selection
    selection_method = selection_method.lower()
    if selection_method == "roulette":
        toolbox.register(alias="select", function=tools.selRoulette)
    elif selection_method == "tournament":
        toolbox.register(alias="select", function=tools.selTournament, tournsize=population_size)
    elif selection_method == "worst":
        toolbox.register(alias="select", function=tools.selWorst)
    elif selection_method == "best":
        toolbox.register(alias="select", function=tools.selBest)
    elif selection_method == "random":
        toolbox.register(alias="select", function=tools.selRandom)
    else:
        raise ValueError()

    # Crossover
    crossover_method = crossover_method.lower()
    if crossover_method == "onepoint":
        toolbox.register(alias="mate", function=crossover_onepoint)
    elif crossover_method == "twopoint":
        toolbox.register(alias="mate", function=crossover_twopoint)
    elif crossover_method == "uniform":
        toolbox.register(alias="mate", function=crossover_uniform)
    elif crossover_method == "pmx":
        toolbox.register(alias="mate", function=crossover_pmx)
    else:
        raise ValueError()

    # Mutation
    mutation_method = mutation_method.lower()
    if mutation_method == "swap":
        toolbox.register(alias="mutate", function=mutation_swap)
    elif mutation_method == "range":
        toolbox.register(alias="mutate", function=mutation_range)
    elif mutation_method == "shuffle":
        toolbox.register(alias="mutate", function=mutation_shuffle)
    else:
        raise ValueError()

    # Replacement
    replacement_method = replacement_method.lower()
    if replacement_method == "parents":
        toolbox.register(alias="replace", function=replacement_parents)
    elif replacement_method == "parents_worse":
        toolbox.register(alias="replace", function=replacement_parents_worse)
    elif replacement_method == "parents_better":
        toolbox.register(alias="replace", function=replacement_parents_better)
    elif replacement_method == "worst":
        toolbox.register(alias="replace", function=replacement_worst)
    elif replacement_method == "best":
        toolbox.register(alias="replace", function=replacement_best)
    else:
        raise ValueError()

    population = toolbox.population(n=population_size)
    halloffame = tools.HallOfFame(population_size, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("mean", np.mean, axis=0)
    stats.register("std", np.std, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "num_evals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_individual: list = [individual for individual in population if not individual.fitness.valid]
    list_fitness = toolbox.map(toolbox.evaluate, invalid_individual)
    for (individual, fitness) in zip(invalid_individual, list_fitness):
        individual.fitness.values = fitness

    if halloffame is not None:
        halloffame.update(population)
        population[:] = halloffame[:]

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, num_evals=len(invalid_individual), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, num_generations + 1):
        # Select the next generation individuals
        parents = toolbox.select(population, k=crossover_size)
        offspring = [toolbox.clone(individual) for individual in parents]

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

            if np.random.random() < mutation_rate:
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
            if np.random.random() < mutation_rate:
                offspring[i], = toolbox.mutate(offspring[i])

        # Evaluate the individuals with an invalid fitness
        invalid_individual = [individual for individual in offspring if not individual.fitness.valid]
        list_fitness = toolbox.map(toolbox.evaluate, invalid_individual)
        for (individual, fitness) in zip(invalid_individual, list_fitness):
            individual.fitness.values = fitness

        # Apply reeplacement
        population = toolbox.replace(population=population, parents=parents, offspring=offspring)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)
            population[:] = halloffame[:]

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, num_evals=len(invalid_individual), **record)
        if verbose:
            print(logbook.stream)

    list_individual: list = list()
    for individual in population:
        temp: dict = {
            "chromosome": individual.tolist(),
            "fitness": individual.fitness.values
        }
        list_individual.append(temp)

    return list_individual, logbook
