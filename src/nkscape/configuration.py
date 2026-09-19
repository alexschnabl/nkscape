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
