"""Shared fixtures and helpers for the genetic algorithm tests."""

import random
from collections.abc import Callable
from typing import Any

import constants
import pytest
from ga_stats import Stats
from genalg import GA

GENES = tuple("ABCDEFGHIJ")


def out_of_place(individual: constants.Individual) -> float:
    """Standardized fitness: the number of genes not in their sorted position."""
    mismatches = sum(a != b for a, b in zip(individual, GENES, strict=False))
    return mismatches + abs(len(individual) - len(GENES))


def random_population(size: int) -> list[constants.Individual]:
    """Random permutations of GENES."""
    return [tuple(random.sample(GENES, len(GENES))) for _ in range(size)]


def default_params(**overrides: Any) -> dict[str, Any]:
    """GA parameters with all termination conditions off unless overridden."""
    params: dict[str, Any] = {
        "population_size": 20,
        "min_length": 1,
        "max_length": None,
        "selection_type": constants.SelectionType.FPS,
        "crossover_type": constants.CrossoverType.OX,
        "crossover_rate": 0.9,
        "mutation_type": constants.MutationType.SWAP | constants.MutationType.REVERSE,
        "mutation_rate": 0.1,
        "max_generations": 10,
        "min_generations": 0,
        "convergence_termination": False,
        "convergence_generations": 5,
        "target_fitness_termination": False,
        "target_fitness": 0,
        "tournament_size": 4,
    }
    params.update(overrides)
    return params


@pytest.fixture(autouse=True)
def seeded_random() -> None:
    """Make every test deterministic; the GA uses the global random module."""
    random.seed(12345)


@pytest.fixture
def make_ga() -> Callable[..., GA]:
    """Factory for a GA over random permutations, with parameter overrides."""

    def factory(**overrides: Any) -> GA:
        params = default_params(**overrides)
        population = random_population(params["population_size"])
        return GA(population, out_of_place, params, Stats())

    return factory
