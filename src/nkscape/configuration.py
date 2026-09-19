"""
Configurations: one value of 0 or 1 per decision.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_configuration(config: ArrayLike, n: int) -> NDArray[np.int_]:
    """
    Turn anything holding N zeros and ones into a checked NumPy row.

    Raises
    ------
    ValueError
        If the configuration does not have N values or contains anything but 0 and 1.
    """
    row = np.asarray(config)
    if row.shape != (n,):
        raise ValueError(
            f"configuration must have {n} values (one per decision), got shape {row.shape}"
        )
    if not ((row == 0) | (row == 1)).all():
        raise ValueError("configuration may only contain 0 and 1")
    return row.astype(int)


def check_configurations(configs: ArrayLike, count: int, n: int) -> NDArray[np.int_]:
    """
    Turn anything holding `count` configurations of N zeros and ones into a checked NumPy grid.

    Raises
    ------
    ValueError
        If there are not `count` rows of N values or anything but 0 and 1 is contained.
    """
    rows = np.asarray(configs)
    if rows.shape != (count, n):
        raise ValueError(
            f"expected {count} configurations of {n} values each, got shape {rows.shape}"
        )
    if not ((rows == 0) | (rows == 1)).all():
        raise ValueError("configurations may only contain 0 and 1")
    return rows.astype(int)
