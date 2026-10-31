"""Tests for ga_stats.Stats."""

import pytest
from ga_stats import Stats

POPULATION = [("A", "B"), ("B", "A"), ("A", "A")]


def test_values_are_none_before_first_update() -> None:
    stats = Stats()
    assert stats.best_fitness is None
    assert stats.best_individual is None
    assert stats.best_generation is None
    assert stats.best_fitness_per_generation == []
    assert stats.convergence_count == 0


def test_update_records_generation_statistics() -> None:
    stats = Stats()
    stats.update([3, 1, 2], POPULATION)
    assert stats.best_fitness == 1
    assert stats.best_individual == ("B", "A")
    assert stats.best_generation == 1
    assert stats.best_fitness_per_generation == [1]
    assert stats.best_individual_per_generation == [("B", "A")]
    assert stats.average_fitness_per_generation == [pytest.approx(2.0)]


def test_best_overall_is_kept_when_a_later_generation_is_worse() -> None:
    stats = Stats()
    stats.update([3, 1, 2], POPULATION)
    stats.update([5, 4, 6], POPULATION)
    assert stats.best_fitness == 1
    assert stats.best_generation == 1
    assert stats.best_fitness_per_generation == [1, 4]


def test_first_generation_does_not_count_as_converged() -> None:
    stats = Stats()
    stats.update([3, 1, 2], POPULATION)
    assert stats.convergence_count == 0


def test_convergence_counts_consecutive_unchanged_generations() -> None:
    stats = Stats()
    for fitness in ([3, 1, 2], [1, 5, 5], [4, 1, 9]):
        stats.update(fitness, POPULATION)
    assert stats.convergence_count == 2


def test_convergence_resets_when_best_fitness_changes() -> None:
    stats = Stats()
    for fitness in ([3, 1, 2], [1, 5, 5], [0, 5, 5]):
        stats.update(fitness, POPULATION)
    assert stats.convergence_count == 0
