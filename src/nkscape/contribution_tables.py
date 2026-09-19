"""
The fitness-contribution tables of an NK landscape.
"""

from collections.abc import Iterable
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from nkscape.configuration import check_configuration
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
        One table per decision, each shaped to fit its row of the matrix. They are taken one at
        a time, so a generator that makes each table only when asked keeps memory low.
    dtype
        How the numbers are stored: `float64` (default, 8 bytes each) or `float32` (4 bytes
        each, about 7 significant digits). Everything computed from the tables - contributions,
        fitness, the fitness cube - uses the same size. `float32` halves the memory; on very
        large landscapes it can make a few neighbouring configurations exactly equal that are
        not equal in `float64`.

    Raises
    ------
    ValueError
        If the number of tables or the shape of a table does not fit the matrix, or `dtype` is
        neither `float64` nor `float32`.
    """

    def __init__(
        self,
        matrix: InteractionMatrix,
        tables: Iterable[ArrayLike],
        dtype: DTypeLike = np.float64,
    ) -> None:
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            raise ValueError(f"dtype must be float64 or float32, got {self.dtype}")

        checked = []
        for i, table in enumerate(tables):
            if i == matrix.n:
                raise ValueError(f"expected {matrix.n} tables (one per decision), got more")
            try:
                array = np.array(table, dtype=self.dtype)
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

        if len(checked) != matrix.n:
            raise ValueError(f"expected {matrix.n} tables (one per decision), got {len(checked)}")

        self.matrix: InteractionMatrix = matrix
        self.tables: tuple[NDArray[np.floating], ...] = tuple(checked)
        self.n: int = matrix.n

    @classmethod
    def random(
        cls,
        matrix: InteractionMatrix,
        rng: int | np.random.Generator,
        dtype: DTypeLike = np.float64,
    ) -> Self:
        """
        Tables filled with random numbers between 0 and 1, drawn from `rng`.

        The numbers are always drawn as `float64` and then stored as `dtype`, so the same seed
        gives the same landscape in either size, only rounded.

        Parameters
        ----------
        matrix
            The interaction matrix the tables belong to.
        rng
            A seed, or a generator to draw from. The same seed always gives the same tables.
        dtype
            `float64` (default) or `float32`; see the class description.
        """
        rng = np.random.default_rng(rng)
        shapes = [(2,) * len(matrix.depends_on(i)) for i in range(matrix.n)]
        return cls(matrix, (rng.random(shape) for shape in shapes), dtype)

    def contributions(self, config: ArrayLike) -> NDArray[np.floating]:
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
        row = check_configuration(config, self.n)
        values = []
        for i in range(self.n):
            answers = tuple(row[self.matrix.depends_on(i)])
            values.append(self.tables[i][answers])
        return np.array(values)

    def contribution_cube(self, i: int) -> NDArray[np.floating]:
        """
        Contribution i of every configuration, as an array with N axes of length 2.

        The position is the configuration: `cube[1, 0, 1]` is contribution i of 101.
        No numbers are copied; the table is only viewed along the decisions it does not
        depend on.
        """
        shape = 2 ** self.matrix.grid[i]
        return np.broadcast_to(self.tables[i].reshape(shape), (2,) * self.n)
