"""
ga_stats.py
Authors: Michael P. Lang

Collects statistics for the genetic algorithm.
"""

from collections.abc import Collection

from constants import Individual


class Stats:
    """Various statistics for the genetic algorithm."""

    def __init__(self):
        """Initialize statistics."""
        # best overall fitness value for entire run of the algorithm
        self._best_fitness = None
        self._best_individual = None
        self._best_generation = None
        # list of best fitness values for each generation
        self._best_fitness_per_generation = []
        self._best_individual_per_generation = []
        # list of average fitness values for each generations
        self._average_fitness_per_generation = []
        self._convergence_count = 0

    def update(
        self, raw_fitness: Collection[int | float], population: list[Individual]
    ) -> None:
        """Collects statistics for the population.

        Parameters
        ----------
        raw_fitness : Collection[Union[int, float]]
            The raw fitness values for the entire population.
        population : list[Individual]
            The current generation's population.
        """
        # Get the best index and individual for this generation.
        best_fitness, index = min([(fit, i) for i, fit in enumerate(raw_fitness)])
        best_individual = population[index]
        average_fitness = sum(raw_fitness) / len(raw_fitness)
        self._best_fitness_per_generation.append(best_fitness)
        self._best_individual_per_generation.append(best_individual)
        self._average_fitness_per_generation.append(average_fitness)

        # Keep track of consecutive generations with the same best fitness values.
        # This is used for convergence testing.
        if (
            len(self._best_fitness_per_generation) > 0
            and best_fitness == self._best_fitness_per_generation[-1]
        ):
            self._convergence_count += 1
        else:
            self._convergence_count = 0

        if self._best_fitness is None or best_fitness < self._best_fitness:
            # Record best overall values
            self._best_fitness = best_fitness
            self._best_individual = best_individual
            self._best_generation = len(self._best_fitness_per_generation)

    @property
    def best_fitness(self) -> float:
        """The best fitness score across all generations."""
        return self._best_fitness

    @property
    def best_individual(self) -> Individual:
        """The best individual across all generations."""
        return self._best_individual

    @property
    def best_generation(self) -> int:
        """The generation that produced the individual with the best fitness."""
        return self._best_generation

    @property
    def best_fitness_per_generation(self) -> list[float]:
        """A list of the best fitness scores seen in each generation."""
        return self._best_fitness_per_generation

    @property
    def best_individual_per_generation(self) -> list[Individual]:
        """A list of the individuals with the best fitness scores seen in each
        generation.
        """
        return self._best_individual_per_generation

    @property
    def average_fitness_per_generation(self) -> list[float]:
        """The mean of the fitness scores across the entire population for each
        generation.
        """
        return self._average_fitness_per_generation

    @property
    def convergence_count(self) -> int:
        """The number of consecutive generations in which the fitness score has not
        changed.
        """
        return self._convergence_count
