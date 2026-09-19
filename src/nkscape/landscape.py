"""
The NK fitness landscape.
"""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.configuration import check_configuration
from nkscape.contribution_tables import ContributionTables
from nkscape.interaction_matrix import InteractionMatrix

# Looking at every configuration holds 2^N fitness values in memory: about 16 bytes per
# configuration while the cube is built, 8 bytes afterwards. N = 24 stays below 0.3 GB.
DEFAULT_ENUMERATION_LIMIT = 24

_enumeration_limit = DEFAULT_ENUMERATION_LIMIT


def get_enumeration_limit() -> int:
    """
    The largest N for which every configuration of a landscape may be looked at.
    """
    return _enumeration_limit


def set_enumeration_limit(n: int) -> None:
    """
    Change the largest N for which every configuration of a landscape may be looked at.

    The limit protects against running out of memory; it never changes a result. Building the
    fitness cube needs about 16 bytes per configuration: 0.3 GB for N = 24, 4.3 GB for N = 28,
    17 GB for N = 30, doubling with every further decision.

    Parameters
    ----------
    n
        The new limit, at least 1. The default is 24.
    """
    global _enumeration_limit
    if n < 1:
        raise ValueError(f"the limit must be at least 1, got {n}")
    _enumeration_limit = n


class Landscape:
    """
    An NK fitness landscape: an interaction matrix together with its contribution tables.

    The fitness of a configuration is the average of its N fitness contributions.

    Parameters
    ----------
    matrix
        The interaction matrix.
    tables
        The contribution tables. They must have been built for `matrix`.

    Raises
    ------
    ValueError
        If the tables were built for a different matrix.
    """

    def __init__(self, matrix: InteractionMatrix, tables: ContributionTables) -> None:
        if tables.matrix is not matrix:
            raise ValueError("the tables were built for a different interaction matrix")
        self.matrix: InteractionMatrix = matrix
        self.tables: ContributionTables = tables
        self.n: int = matrix.n
        self._fitness_cube: NDArray[np.float64] | None = None
        self._flips: NDArray[np.int_] = np.eye(self.n, dtype=int)
        self._flips.setflags(write=False)

    @classmethod
    def with_random_tables(cls, matrix: InteractionMatrix, rng: int | np.random.Generator) -> Self:
        """
        Landscape for a given interaction matrix, with random contribution tables.

        Parameters
        ----------
        matrix
            The interaction matrix.
        rng
            A seed, or a generator to draw from. The same seed always gives the same tables.
        """
        return cls(matrix, ContributionTables.random(matrix, rng))

    @classmethod
    def random_k(cls, n: int, k: int, rng: int | np.random.Generator) -> Self:
        """
        Landscape with a random-K interaction matrix and random contribution tables.

        The matrix and the tables draw from separate streams spawned from `rng`.

        Parameters
        ----------
        n, k
            Number of decisions, and number of other decisions per contribution.
        rng
            A seed, or a generator to draw from. The same seed always gives the same landscape.
        """
        matrix_rng, tables_rng = np.random.default_rng(rng).spawn(2)
        return cls.with_random_tables(InteractionMatrix.random_k(n, k, matrix_rng), tables_rng)

    @classmethod
    def next_k(cls, n: int, k: int, rng: int | np.random.Generator) -> Self:
        """
        Landscape with a next-K interaction matrix and random contribution tables.

        Parameters
        ----------
        n, k
            Number of decisions, and number of other decisions per contribution.
        rng
            A seed, or a generator to draw from. The same seed always gives the same landscape.
        """
        return cls.with_random_tables(InteractionMatrix.next_k(n, k), rng)

    def contributions(self, config: ArrayLike) -> NDArray[np.float64]:
        """
        The fitness contributions of one configuration, one value per decision.
        """
        return self.tables.contributions(config)

    def fitness(self, config: ArrayLike) -> float:
        """
        The fitness of one configuration: the average of its contributions.

        Read from the fitness cube if that has been built, otherwise looked up.
        """
        row = check_configuration(config, self.n)
        if self._fitness_cube is not None:
            return float(self._fitness_cube[tuple(row)])
        total = 0.0
        for value in self.tables.contributions(row):
            total += value
        return float(total / self.n)

    def neighbours(self, config: ArrayLike) -> NDArray[np.int_]:
        """
        The N configurations that differ from `config` in one decision.

        Row j is `config` with decision j flipped.
        """
        row = check_configuration(config, self.n)
        return row ^ self._flips

    def neighbour_fitness(self, config: ArrayLike) -> NDArray[np.float64]:
        """
        The fitness of every neighbour: entry j is the fitness after flipping decision j.

        Read from the fitness cube if that has been built, otherwise looked up.
        """
        neighbours = self.neighbours(config)
        if self._fitness_cube is not None:
            from_cube: NDArray[np.float64] = self._fitness_cube[tuple(neighbours.T)]
            return from_cube
        return np.array([self.fitness(row) for row in neighbours])

    def is_peak(self, config: ArrayLike) -> bool:
        """
        Whether no single-decision change improves fitness (ties count as a peak).
        """
        return bool((self.neighbour_fitness(config) <= self.fitness(config)).all())

    def random_configuration(self, rng: int | np.random.Generator) -> NDArray[np.int_]:
        """
        A configuration drawn uniformly at random.
        """
        return np.random.default_rng(rng).integers(2, size=self.n)

    def fitness_cube(self) -> NDArray[np.float64]:
        """
        The fitness of every configuration, as an array with N axes of length 2.

        The position is the configuration: `cube[1, 0, 1]` is the fitness of 101.
        Computed on first use and then kept.

        Raises
        ------
        ValueError
            If N is above the enumeration limit (24 unless changed with
            `set_enumeration_limit`), because the array holds all 2^N fitness values.
        """
        if self._fitness_cube is None:
            if self.n > _enumeration_limit:
                raise ValueError(
                    f"looking at every configuration needs 2^N fitness values; N = {self.n} "
                    f"is above the limit of {_enumeration_limit}. Building the cube for "
                    f"N = {self.n} takes about {16 * 2**self.n / 1e9:.1f} GB of memory; if this "
                    f"machine has that, raise the limit with nk.set_enumeration_limit({self.n})"
                )
            total = np.zeros((2,) * self.n)
            for i in range(self.n):
                total += self.tables.contribution_cube(i)
            cube = total / self.n
            cube.setflags(write=False)
            self._fitness_cube = cube
        return self._fitness_cube

    def peaks(self) -> NDArray[np.intp]:
        """
        The local optima: configurations that no single-decision change improves.

        One row per peak. A configuration whose best neighbour is exactly as good counts as a peak.
        """
        cube = self.fitness_cube()
        is_peak = np.ones(cube.shape, dtype=bool)
        for j in range(self.n):
            is_peak &= cube >= np.flip(cube, axis=j)
        return np.argwhere(is_peak)

    def count_peaks(self) -> int:
        """
        The number of local optima.
        """
        return len(self.peaks())

    def global_max(self) -> float:
        """
        The highest fitness on the landscape.
        """
        return float(self.fitness_cube().max())
