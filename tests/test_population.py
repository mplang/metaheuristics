"""Tests for population.Population."""

from collections import Counter

import constants
import pytest
from conftest import GENES, out_of_place, random_population
from ga_stats import Stats
from population import Population

TRIALS = 200


def make_population(**kwargs) -> Population:
    return Population(random_population(10), out_of_place, **kwargs)


def test_fitness_is_a_probability_distribution() -> None:
    population = make_population()
    assert len(population.fitness) == len(population.population)
    assert all(0 < fit <= 1 for fit in population.fitness)
    assert sum(population.fitness) == pytest.approx(1.0)
    assert population.total_fitness == pytest.approx(1.0)


def test_sum_fitness_is_cumulative() -> None:
    population = make_population()
    running = 0.0
    for fit, cumulative in zip(population.fitness, population.sum_fitness, strict=True):
        running += fit
        assert cumulative == pytest.approx(running)


def test_lower_raw_fitness_gets_higher_selection_probability() -> None:
    best, worst = GENES, GENES[::-1]
    population = Population([best, worst], out_of_place)
    assert population.fitness[0] > population.fitness[1]


def test_evaluation_updates_stats() -> None:
    stats = Stats()
    Population([GENES, GENES[::-1]], out_of_place, stats=stats)
    assert stats.best_fitness == 0
    assert stats.best_individual == GENES


def test_without_stats_convergence_is_zero_and_display_is_silent(capsys) -> None:
    population = make_population()
    assert population.convergence_count == 0
    population.display_stats()
    assert capsys.readouterr().out == ""


def test_clear_population_empties_population_and_fitness() -> None:
    population = make_population()
    population.clear_population()
    assert population.population == []
    assert population.fitness == []
    assert population.total_fitness == 0.0


@pytest.mark.parametrize(
    "mutation_type", [constants.MutationType.SWAP, constants.MutationType.REVERSE]
)
def test_reordering_mutations_preserve_genes(mutation_type) -> None:
    population = make_population()
    for _ in range(TRIALS):
        mutated = population.mutate(GENES, mutation_type, mutation_rate=1.0)
        assert isinstance(mutated, tuple)
        assert Counter(mutated) == Counter(GENES)


def test_replicate_adds_one_existing_gene_up_to_max_length() -> None:
    population = make_population(max_length=len(GENES) + 1)
    replicate = constants.MutationType.REPLICATE
    for _ in range(TRIALS):
        mutated = population.mutate(GENES, replicate, mutation_rate=1.0)
        assert len(mutated) == len(GENES) + 1
        assert set(mutated) == set(GENES)
        assert population.mutate(mutated, replicate, mutation_rate=1.0) == mutated


def test_delete_removes_one_gene_down_to_min_length() -> None:
    population = make_population(min_length=len(GENES) - 1)
    delete = constants.MutationType.DELETE
    for _ in range(TRIALS):
        mutated = population.mutate(GENES, delete, mutation_rate=1.0)
        assert len(mutated) == len(GENES) - 1
        assert set(mutated) < set(GENES)
        assert population.mutate(mutated, delete, mutation_rate=1.0) == mutated


def test_zero_mutation_rate_leaves_individual_unchanged() -> None:
    population = make_population()
    assert population.mutate(GENES, constants.MutationType.ALL, 0.0) == GENES


def test_custom_mutation_is_applied() -> None:
    population = make_population()
    population.add_custom_mutation(lambda individual: individual[::-1])
    mutated = population.mutate(GENES, constants.MutationType.CUSTOM, 1.0)
    assert mutated == GENES[::-1]
