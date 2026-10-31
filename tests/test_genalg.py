"""Tests for genalg.GA."""

import random
from collections import Counter

import constants
import pytest
from conftest import GENES, default_params, random_population
from ga_stats import Stats
from genalg import GA

TRIALS = 200

PERMUTATION_CROSSOVERS = [
    constants.CrossoverType.OX,
    constants.CrossoverType.PMX,
    constants.CrossoverType.ERX,
]


def random_permutation() -> tuple[str, ...]:
    return tuple(random.sample(GENES, len(GENES)))


@pytest.mark.parametrize("crossover_type", PERMUTATION_CROSSOVERS)
def test_permutation_crossovers_produce_two_valid_permutations(
    make_ga, crossover_type
) -> None:
    ga = make_ga(crossover_type=crossover_type, crossover_rate=1.0)
    for _ in range(TRIALS):
        children = ga._crossover(random_permutation(), random_permutation())
        assert len(children) == 2
        for child in children:
            assert isinstance(child, tuple)
            assert Counter(child) == Counter(GENES)


def test_single_point_crossover_swaps_tails(make_ga) -> None:
    ga = make_ga()
    parent1, parent2 = tuple("AAAAAA"), tuple("BBBBBB")
    for _ in range(TRIALS):
        child1, child2 = ga._single_point_crossover(parent1, parent2)
        point = child1.index("B")
        assert child1 == parent1[:point] + parent2[point:]
        assert child2 == parent2[:point] + parent1[point:]


def test_no_crossover_returns_parents(make_ga) -> None:
    ga = make_ga(crossover_rate=0.0)
    parent1, parent2 = random_permutation(), random_permutation()
    assert ga._crossover(parent1, parent2) == (parent1, parent2)


@pytest.mark.parametrize("selection_type", list(constants.SelectionType))
def test_selection_returns_pool_of_existing_individuals(
    make_ga, selection_type
) -> None:
    ga = make_ga(selection_type=selection_type)
    pool = ga._get_mating_pool(30, selection_type)
    assert len(pool) == 30
    assert all(individual in ga.population.population for individual in pool)


def test_tournament_selection_with_full_tournament_picks_the_best(make_ga) -> None:
    ga = make_ga(population_size=10, tournament_size=10)
    fitness = ga.population.fitness
    best = ga.population.population[fitness.index(max(fitness))]
    assert ga._ts(5) == [best] * 5


@pytest.mark.parametrize("population_size", [20, 21])
@pytest.mark.parametrize("crossover_type", PERMUTATION_CROSSOVERS)
@pytest.mark.parametrize("selection_type", list(constants.SelectionType))
def test_population_size_is_constant_across_generations(
    make_ga, selection_type, crossover_type, population_size
) -> None:
    ga = make_ga(
        selection_type=selection_type,
        crossover_type=crossover_type,
        population_size=population_size,
    )
    sizes = []
    replace = ga.population.replace_and_evaluate

    def record_size(population) -> None:
        sizes.append(len(population))
        replace(population)

    ga.population.replace_and_evaluate = record_size
    ga.evolve(display_progress=False)
    assert sizes == [population_size] * ga.params["max_generations"]


def test_evolve_stops_at_target_fitness(make_ga, capsys) -> None:
    ga = make_ga(
        max_generations=200,
        target_fitness_termination=True,
        target_fitness=0,
        selection_type=constants.SelectionType.TS,
    )
    ga.evolve(display_progress=False)
    assert ga._stats.best_fitness == 0
    assert "Target fitness achieved" in capsys.readouterr().out


def test_evolve_stops_after_convergence_generations_without_change(capsys) -> None:
    # With constant fitness the best never changes, so the run converges once the
    # unchanged streak exceeds convergence_generations.
    params = default_params(
        max_generations=50, convergence_termination=True, convergence_generations=5
    )
    population = random_population(params["population_size"])
    ga = GA(population, lambda individual: 1, params, Stats())
    ga.evolve(display_progress=False)
    assert "Converged in 5 generations" in capsys.readouterr().out
    assert len(ga._stats.best_fitness_per_generation) == 7  # initial + 6 generations
