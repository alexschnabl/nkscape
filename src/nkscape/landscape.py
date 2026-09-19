"""
The NK fitness landscape.
"""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.contribution_tables import ContributionTables
from nkscape.interaction_matrix import InteractionMatrix


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
        """
        return float(self.contributions(config).mean())
