"""
genalg.py
Author: Michael P. Lang

Genetic Algorithm algorithms and utilities.
Depends on Python3, ProgressBar, and pathos.
"""

# standard library modules
import bisect
import itertools
import random
from collections.abc import Callable, Collection
from typing import Any

# custom modules
import constants
from ga_stats import Stats

# 3rd-party modules
from pathos.multiprocessing import ProcessPool
from population import Population
from progressbar import Bar, Percentage, ProgressBar


class GA:
    """Genetic Algorithm."""

    def __init__(
        self,
        population: Collection[constants.Individual],
        eval_func: Callable[[Collection[str]], int | float],
        params: dict[str, Any],
        stats: Stats,
    ) -> None:
        """Initialize the genetic algorithm.

        Parameters
        ----------
        population : Collection[constants.Individual]
            The initial population.
        eval_func : Callable[[constants.Individual], Union[int, float]]
            The fitness evaluation function.
        params : dict[str, Any]
            Runtime configuration parameters.
        stats : Stats
            Record-keeping.
        """
        self.params = params
        self._stats = stats
        self.population = Population(
            population,
            eval_func,
            stats=stats,
            min_length=self.params["min_length"],
            max_length=self.params["max_length"],
        )

    def _get_mating_pool_size(self) -> int:
        """Return the required size of the mating pool based on the selection type
        and population size.

        Returns
        -------
        int
            The size of the mating pool.
        """
        selection_type = self.params["selection_type"]
        population_size = self.params["population_size"]
        if selection_type == constants.SelectionType.TS:
            # for selection types which return one child,
            # we need to double the size of the mating pool
            mating_pool_size = population_size * 2
        else:
            # selection types which return two children need
            # exactly as many parents as children
            mating_pool_size = population_size

        return mating_pool_size

    def _get_mating_pool(
        self, pool_size: int, selection_type: constants.SelectionType
    ) -> Collection[constants.Individual]:
        """Get the mating pool based on the selection type.

        Parameters
        ----------
        pool_size : int
            The size of the mating pool.
        selection_type : constants.SelectionType
            The mating pool selection strategy.

        Returns
        -------
        Collection[constants.Individual]
            The individuals selected for mating.

        Raises
        ------
        Exception
            An invalid selection type was specified.
        """
        if selection_type == constants.SelectionType.SUS:
            # stochastic universal sampling
            mating_pool = self._sus(pool_size)
        elif selection_type == constants.SelectionType.TS:
            # tournament selection
            mating_pool = self._ts(pool_size)
        elif selection_type == constants.SelectionType.FPS:
            # fitness proportionate selection
            mating_pool = self._fps(pool_size)
        else:
            raise Exception(f"Invalid selection type: '{selection_type}'")

        return mating_pool

    def _single_point_crossover(
        self, parent1: constants.Individual, parent2: constants.Individual
    ) -> tuple[constants.Individual, constants.Individual]:
        """Single-point order crossover.

        Parameters
        ----------
        parent1 : constants.Individual
            The first parent.
        parent2 : constants.Individual
            The second parent.

        Returns
        -------
        tuple[constants.Individual, constants.Individual]
            The two offspring.
        """
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def _crossover_order(
        self, parent1: constants.Individual, parent2: constants.Individual
    ) -> tuple[constants.Individual, constants.Individual]:
        """Two-point order crossover.

        Parameters
        ----------
        parent1 : constants.Individual
            The first parent.
        parent2 : constants.Individual
            The second parent.

        Returns
        -------
        tuple[constants.Individual, constants.Individual]
            The two offspring.
        """
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(crossover_point1, len(parent1))
        # Get a list of items in parent2, starting from crossover_point2, which
        # are not in the middle segment of parent1.
        unused = tuple(
            [
                x
                for x in parent2[crossover_point2:] + parent2[:crossover_point2]
                if x not in parent1[crossover_point1:crossover_point2]
            ]
        )
        # Copy the middle segment from parent1 to child1, and fill in the empty
        # slots from the unused list, beginning with crossover_point2 and
        # wrapping around to the beginning.
        child1 = (
            unused[len(parent1) - crossover_point2 :]
            + parent1[crossover_point1:crossover_point2]
            + unused[: len(parent1) - crossover_point2]
        )

        # Get a list of items in parent1, starting from crossover_point2, which
        # are not in the middle segment of parent2.
        unused = tuple(
            [
                x
                for x in parent1[crossover_point2:] + parent1[:crossover_point2]
                if x not in parent2[crossover_point1:crossover_point2]
            ]
        )
        # Copy the middle segment from parent1 to child1, and fill in the empty
        # slots from the unused list, beginning with crossover_point2 and
        # wrapping around to the beginning.
        child2 = (
            unused[len(parent1) - crossover_point2 :]
            + parent2[crossover_point1:crossover_point2]
            + unused[: len(parent1) - crossover_point2]
        )

        return child1, child2

    def _crossover_partially_mapped(
        self, parent1: constants.Individual, parent2: constants.Individual
    ) -> tuple[constants.Individual, constants.Individual]:
        """Partially-mapped crossover.

        Parameters
        ----------
        parent1 : constants.Individual
            The first parent.
        parent2 : constants.Individual
            The second parent.

        Returns
        -------
        tuple[constants.Individual, constants.Individual]
            The two offspring.
        """
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1))

        child1 = [None] * len(parent1)
        child1[crossover_point1:crossover_point2] = parent1[
            crossover_point1:crossover_point2
        ]
        for pos in range(crossover_point1, crossover_point2):
            i = parent2[pos]
            if i not in parent1[crossover_point1:crossover_point2]:
                j = parent1[pos]
                pos = parent2.index(j)
                while child1[pos]:
                    j = parent1[pos]
                    pos = parent2.index(j)
                child1[pos] = i

        unused = [i for i in parent2 if i not in child1]
        pos = 0
        for i in range(len(child1)):
            if child1[i] is None:
                child1[i] = unused[pos]
                pos += 1

        child2 = [None] * len(parent1)
        child2[crossover_point1:crossover_point2] = parent2[
            crossover_point1:crossover_point2
        ]
        for pos in range(crossover_point1, crossover_point2):
            i = parent1[pos]
            if i not in parent2[crossover_point1:crossover_point2]:
                j = parent2[pos]
                pos = parent1.index(j)
                while child2[pos]:
                    j = parent2[pos]
                    pos = parent1.index(j)
                child2[pos] = i

        unused = [i for i in parent1 if i not in child2]
        pos = 0
        for i in range(len(child2)):
            if child2[i] is None:
                child2[i] = unused[pos]
                pos += 1

        return child1, child2

    def _crossover_edge_recombination(
        self, parent1: constants.Individual, parent2: constants.Individual
    ) -> tuple[constants.Individual, ...]:
        """Edge recombination crossover.

        Parameters
        ----------
        parent1 : constants.Individual
            The first parent.
        parent2 : constants.Individual
            The second parent.

        Returns
        -------
        tuple[constants.Individual, ...]
            The offspring.
        """
        adj1 = {
            parent1[i]: {parent1[i - 1], parent1[(i + 1) % len(parent1)]}
            for i in range(len(parent1))
        }
        adj2 = {
            parent2[i]: {parent2[i - 1], parent2[(i + 1) % len(parent2)]}
            for i in range(len(parent2))
        }
        adjset = {
            parent1[i]: adj1[parent1[i]].union(adj2[parent1[i]])
            for i in range(len(parent1))
        }

        child = []
        n = random.choice([parent1[0], parent2[0]])
        while True:
            child.append(n)

            if len(child) == len(parent1):
                break

            for node in adjset:
                if n in adjset[node]:
                    adjset[node].remove(n)

            if len(adjset[n]) != 0:
                if len(adjset[n]) == 1:
                    nxt = adjset[n].pop()
                else:
                    lengths = {
                        neighbor: len(adjset[neighbor]) for neighbor in adjset[n]
                    }
                    # next = sorted(lengths, key=lengths.get)[0]
                    nxt = random.choice(
                        [c for c in lengths if lengths[c] == min(lengths.values())]
                    )
            else:
                nxt = n
                while nxt in child:
                    nxt = random.choice(parent1)
            n = nxt

        return [child]

    def _crossover(
        self, parent1: constants.Individual, parent2: constants.Individual
    ) -> tuple[constants.Individual, ...]:
        """Mate two individuals to create offsprint.

        Parameters
        ----------
        parent1 : constants.Individual
            The first parent.
        parent2 : constants.Individual
            The second parent.

        Returns
        -------
        tuple[constants.Individual, ...]
            The offspring.
        """
        if random.random() > self.params["crossover_rate"]:
            return parent1, parent2
        else:
            crossover_type = self.params["crossover_type"]
            if crossover_type == constants.CrossoverType.OX:
                # ordered crossover
                return self._crossover_order(parent1, parent2)
            elif crossover_type == constants.CrossoverType.PMX:
                # partially-mapped crossover
                return self._crossover_partially_mapped(parent1, parent2)
            elif crossover_type == constants.CrossoverType.ERX:
                # edge recombination crossover
                return self._crossover_edge_recombination(parent1, parent2)
            else:
                return parent1, parent2

    def _fps(self, mating_pool_size: int) -> list[constants.Individual]:
        """Generate the mating pool using fitness proportionate selection.

        Parameters
        ----------
        mating_pool_size : int
            The size of the mating pool.

        Returns
        -------
        list[constants.Individual]
            constants.Individuals selected for recombination.
        """
        # Conceptially, this operation is akin to spinning a game wheel
        # where each member of the population is represented by a slice
        # of the wheel proportionate to its fitness. This is repeated once
        # for each spot in the mating pool.
        mating_pool = []
        total_fitness = self.population.total_fitness
        for _i in range(mating_pool_size):
            # Generate a random value between 0 and the cumulative fitness.
            # When using the normalized fitness, this will be between 0 and 1.
            rand = random.random() * total_fitness
            # the bisect operation will select the first item greater-than rand
            mating_pool.append(
                self.population.population[
                    bisect.bisect(self.population.sum_fitness, rand)
                ]
            )

        return mating_pool

    def _sus(self, mating_pool_size: int) -> list[constants.Individual]:
        """Generate the mating pool using stochastic universal sampling.

        Parameters
        ----------
        mating_pool_size : int
            The size of the mating pool.

        Returns
        -------
        list[constants.Individual]
            constants.Individuals selected for recombination.
        """
        mating_pool = []
        p_dist = 1.0 / (mating_pool_size)
        rand = random.uniform(0, p_dist)
        points = [rand + i * p_dist for i in range(mating_pool_size)]
        i = 0
        for p in points:
            while self.population.sum_fitness[i] < p:
                i += 1
            mating_pool.append(self.population.population[i]["individual"])

        return mating_pool

    def _ts(self, mating_pool_size: int) -> list[constants.Individual]:
        """Generate the mating pool using tournament selection.

        Parameters
        ----------
        mating_pool_size : int
            The size of the mating pool.

        Returns
        -------
        list[constants.Individual]
            constants.Individuals selected for recombination.
        """
        mating_pool = []
        population_size = self.params["population_size"]
        for i in range(mating_pool_size):
            tournament_size = self.params["tournament_size"]
            contenders = random.sample(range(population_size), tournament_size)
            fit = self.population.fitness[contenders[0]]
            parent = self.population.population[contenders[0]]["individual"]
            for i in range(1, tournament_size):
                if self.population.fitness[contenders[i]] > fit:
                    fit = self.population.fitness[contenders[i]]
                    parent = self.population.population[contenders[i]]["individual"]
            mating_pool.append(parent)

        return mating_pool

    def _mate_and_mutate(
        self,
        parent1: constants.Individual,
        parent2: constants.Individual,
        mutation_type: constants.MutationType,
        mutation_rate: float,
    ) -> list[constants.Individual]:
        """Performs recombination of two parents and randomly mutates the children.

        Parameters
        ----------
        parent1 : constants.Individual
            The first parent.
        parent2 : constants.Individual
            The second parent.
        mutation_type : constants.MutationType
            The mutation strategy or strategies to apply.
        mutation_rate : float
            The rate of mutation for each child and strategy.

        Returns
        -------
        list[constants.Individual]
            _description_
        """
        return [
            self.population.mutate(child, mutation_type, mutation_rate)
            for child in self._crossover(parent1, parent2)
        ]

    def add_custom_mutation(
        self, mutate_function: Callable[[constants.Individual], constants.Individual]
    ) -> None:
        """Add a custom mutation function to the available mutation strategies.
        This function will be a candidate for execution when the mutation type is
        CUSTOM or ALL.

        Parameters
        ----------
        mutate_function : Callable[[constants.Individual], constants.Individual]
            A function implementing a custom mutation strategy.
        """
        self.population.add_custom_mutation(mutate_function)

    def evolve(self, display_progress: bool = True) -> None:
        """Run the genetic algorithm.

        Parameters
        ----------
        display_progress : bool, optional
            Whether or not to display a progress bar in the console, by default True.
        """
        # grab some settings to avoid repeated lookups
        max_generations = self.params["max_generations"]
        mutation_type = self.params["mutation_type"]
        mutation_rate = self.params["mutation_rate"]
        mating_pool_size = self._get_mating_pool_size()
        selection_type = self.params["selection_type"]
        convergence_termination = (
            self.params["convergence_generations"]
            if self.params["convergence_termination"]
            else None
        )
        min_generations = self.params["min_generations"]
        fitness_termination = (
            self.params["target_fitness"]
            if self.params["target_fitness_termination"]
            else None
        )

        if display_progress:
            pbar = ProgressBar(
                widgets=[Percentage(), Bar()], maxval=max_generations
            ).start()
        else:
            pbar = None

        for generation in range(max_generations):
            if not display_progress:
                print(f"Generation {generation}: ", end="")
            mating_pool = self._get_mating_pool(
                mating_pool_size, selection_type
            )  # selection
            results = []
            # Disabling for now; the overhead of multiprocessing is MUCH higher than
            # the processing cost.
            do_parallel = False
            if do_parallel:
                with ProcessPool() as pool:
                    for i in range(0, len(mating_pool), 2):
                        # pairwise recombination
                        # TODO: mating strategies for > 2 parents
                        parent1 = mating_pool[i]
                        parent2 = mating_pool[i + 1]
                        results.append(
                            pool.apipe(
                                self._mate_and_mutate,
                                parent1,
                                parent2,
                                mutation_type,
                                mutation_rate,
                            )
                        )
                # Wait for mating to complete and collect the results in a list
                self.population.replace_and_evaluate(
                    list(itertools.chain(*[res.get() for res in results]))
                )
            else:
                population = []
                for i in range(0, len(mating_pool), 2):
                    # pairwise recombination
                    # TODO: mating strategies for > 2 parents
                    parent1 = mating_pool[i]
                    parent2 = mating_pool[i + 1]
                    population.extend(
                        self._mate_and_mutate(
                            parent1, parent2, mutation_type, mutation_rate
                        )
                    )
                self.population.replace_and_evaluate(population)
                if not display_progress:
                    self.population.display_stats()
            if convergence_termination is not None and generation > min_generations:
                if self.population.convergence_count > convergence_termination:
                    print(f"Converged in {generation} generations")
                    break
            if (
                fitness_termination is not None
                and self._stats.best_fitness == fitness_termination
            ):
                print(f"Target fitness achieved in {generation} generations")
                break
            if pbar is not None:
                pbar.update(generation + 1)
        if pbar is not None:
            pbar.finish()
