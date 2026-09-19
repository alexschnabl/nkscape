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

    def contributions(self, config: ArrayLike) -> NDArray[np.float64]:
        """
        The fitness contributions of one configuration, one value per decision.

        Parameters
        ----------
        config
            The configuration: N values of 0 or 1, one per decision.

        Raises
        ------
        ValueError
            If the configuration does not have N values or contains anything but 0 and 1.
        """
        row = np.asarray(config)
        if row.shape != (self.n,):
            raise ValueError(
                f"configuration must have {self.n} values (one per decision), got shape {row.shape}"
            )
        if not np.isin(row, [0, 1]).all():
            raise ValueError("configuration may only contain 0 and 1")
        row = row.astype(int)

        values = []
        for i in range(self.n):
            answers = tuple(row[self.matrix.depends_on(i)])
            values.append(self.tables[i][answers])
        return np.array(values)

    def contribution_cube(self, i: int) -> NDArray[np.float64]:
        """
        Contribution i of every configuration, as an array with N axes of length 2.

        The position is the configuration: `cube[1, 0, 1]` is contribution i of 101.
        No numbers are copied; the table is only viewed along the decisions it does not
        depend on.
        """
        shape = [2 if x else 1 for x in self.matrix.grid[i]]
        return np.broadcast_to(self.tables[i].reshape(shape), (2,) * self.n)
