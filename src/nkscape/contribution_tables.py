"""
The fitness-contribution tables of an NK landscape.
"""

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.interaction_matrix import InteractionMatrix


class ContributionTables:
    """
    Fitness-contribution tables of an NK landscape, one table per decision.

    Table i holds the contribution of decision i for every combination of the decisions it
    depends on. It has one axis of length 2 per decision in `matrix.depends_on(i)`, in that
    order.

    The tables cannot be changed after creation; build new ones instead.

    Parameters
    ----------
    matrix
        The interaction matrix the tables belong to.
    tables
        One table per decision, each shaped to fit its row of the matrix.

    Raises
    ------
    ValueError
        If the number of tables or the shape of a table does not fit the matrix.
    """

    def __init__(self, matrix: InteractionMatrix, tables: Sequence[ArrayLike]) -> None:
        if len(tables) != matrix.n:
            raise ValueError(f"expected {matrix.n} tables (one per decision), got {len(tables)}")

        checked = []
        for i, table in enumerate(tables):
            try:
                array = np.array(table, dtype=float)
            except ValueError:
                raise ValueError(f"table {i} is not a regular grid of numbers") from None

            expected = (2,) * len(matrix.depends_on(i))
            if array.shape != expected:
                raise ValueError(
                    f"table {i} must have shape {expected} because contribution {i} depends "
                    f"on decisions {matrix.depends_on(i)}, got shape {array.shape}"
                )

            array.setflags(write=False)
            checked.append(array)

        self.matrix: InteractionMatrix = matrix
        self.tables: tuple[NDArray[np.float64], ...] = tuple(checked)
        self.n: int = matrix.n

    @classmethod
    def random(cls, matrix: InteractionMatrix, rng: int | np.random.Generator) -> Self:
        """
        Tables filled with random numbers between 0 and 1, drawn from `rng`.

        Parameters
        ----------
        matrix
            The interaction matrix the tables belong to.
        rng
            A seed, or a generator to draw from. The same seed always gives the same tables.
        """
        rng = np.random.default_rng(rng)
        shapes = [(2,) * len(matrix.depends_on(i)) for i in range(matrix.n)]
        return cls(matrix, [rng.random(shape) for shape in shapes])
