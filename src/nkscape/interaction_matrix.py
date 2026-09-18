"""
The interaction matrix of an NK landscape.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class InteractionMatrix:
    """
    Interaction matrix of an NK landscape.

    Entry (i, j) is 1 if the fitness contribution of decision i depends on decision j.
    Row i therefore marks the decisions that contribution i depends on, and column j marks
    the contributions that change when decision j is flipped. The diagonal is always 1,
    because a contribution always depends on its own decision.

    The matrix cannot be changed after creation; build a new one instead.

    Parameters
    ----------
    rows
        The N rows of the matrix, each holding N values of 0 or 1.

    Raises
    ------
    ValueError
        If the rows do not form a square matrix, contain values other than 0 and 1,
        or the diagonal is not all 1.
    """

    def __init__(self, rows: Sequence[Sequence[int]] | NDArray[np.int_]) -> None:
        try:
            grid = np.asarray(rows)
        except ValueError:
            raise ValueError("all rows must have the same length") from None

        if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
            raise ValueError(f"matrix must be square (N rows, N columns), got shape {grid.shape}")

        if not np.isin(grid, [0, 1]).all():
            raise ValueError("matrix may only contain 0 and 1")

        missing = [i for i in range(len(grid)) if grid[i, i] != 1]
        if missing:
            raise ValueError(
                "the diagonal must be all 1 (a contribution always depends on its own "
                f"decision); it is not 1 in rows {missing}"
            )

        self.grid: NDArray[np.int_] = grid.astype(int)
        self.grid.setflags(write=False)
        self.n: int = len(grid)

    def __str__(self) -> str:
        """
        The matrix as text: X where contribution i depends on decision j, K at the end of each row.
        """
        width = len(f"d{self.n - 1}")
        header = " ".join(f"{f'd{j}':>{width}}" for j in range(self.n))
        lines = [" " * width + " " + header]
        for i in range(self.n):
            cells = " ".join(f"{'X' if x else '.':>{width}}" for x in self.grid[i])
            lines.append(f"{f'c{i}':>{width}} {cells}   K={self.k_per_row[i]}")
        return "\n".join(lines)

    @property
    def k_per_row(self) -> NDArray[np.int_]:
        """
        K of each row: how many other decisions contribution i depends on.
        """
        return self.grid.sum(axis=1) - 1

    @property
    def k(self) -> int:
        """
        The K shared by all rows.

        Raises
        ------
        ValueError
            If the rows do not all have the same K. Use `k_per_row` in that case.
        """
        ks = self.k_per_row
        if not (ks == ks[0]).all():
            raise ValueError(f"rows have different K ({ks}); use k_per_row instead")
        return int(ks[0])

    def depends_on(self, i: int) -> NDArray[np.intp]:
        """
        Decisions that contribution i depends on (row i), in ascending order.

        Always includes i itself.
        """
        return np.flatnonzero(self.grid[i])

    def affects(self, j: int) -> NDArray[np.intp]:
        """
        Contributions that change when decision j is flipped (column j), in ascending order.

        Always includes j itself.
        """
        return np.flatnonzero(self.grid[:, j])
