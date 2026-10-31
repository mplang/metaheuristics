"""
constants.py
Authors: Michael P. Lang

Constants and enumerations defined for use by the genetic algorithm.
"""

from enum import Flag, IntEnum, auto

Individual = tuple[str, ...]  # typedef representing a population individual


class OptimizationType(IntEnum):
    """Whether to optimize for minimum or maximum fitness values.

    Attributes
    ----------
    MIN
        Select for maximum fitness values
    MAX
        Select for minimum fitness values
    """

    MIN = 0
    MAX = 1


class SelectionType(IntEnum):
    """The selection strategy for chosing individuals for the mating pool.

    Attributes
    ----------
    SUS
        Stochastic universal sampling
    FPS
        Fitness proportionate selection
    TS
        Tournament selection
    """

    SUS = 0
    FPS = 1
    TS = 2


class CrossoverType(IntEnum):
    """The crossover/mating strategy.

    Attributes
    ----------
    OX
        2-point ordered crossover
    PMX
        Partially-mapped crossover
    ERX
        Edge recombination crossover
    SPX
        Single-point crossover
    """

    OX = 0
    PMX = 1
    ERX = 2
    SPX = 3


class MutationType(Flag):
    """The available mutation strategies.

    Attributes
    ----------
    SWAP
        Swap two genes
    REVERSE
        Reverse a segment
    DELETE
        Delete a gene
    REPLICATE
        Replicate a gene
    CUSTOM
        Custom-defined mutation function(s)
    ALL
        Allow any defined mutation
    """

    SWAP = auto()
    REVERSE = auto()
    DELETE = auto()
    REPLICATE = auto()
    CUSTOM = auto()
    ALL = SWAP | REVERSE | DELETE | REPLICATE | CUSTOM
