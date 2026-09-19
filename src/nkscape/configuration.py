"""
Configurations: one value of 0 or 1 per decision.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_configuration(config: ArrayLike, n: int, count: int | None = None) -> NDArray[np.int_]:
    """
    Turn anything holding zeros and ones into a checked NumPy array.

    Without `count`, one configuration of N values is expected; with it, `count` configurations,
    one per row.

    Raises
    ------
    ValueError
        If the shape is not as expected or anything but 0 and 1 is contained.
    """
    rows = np.asarray(config)
    if rows.shape != ((n,) if count is None else (count, n)):
        expected = f"{n} values" if count is None else f"{count} configurations of {n} values"
        raise ValueError(f"expected {expected} (one per decision), got shape {rows.shape}")
    if not ((rows == 0) | (rows == 1)).all():
        raise ValueError("a configuration may only contain 0 and 1")
    checked: NDArray[np.int_] = rows.astype(int)
    return checked
