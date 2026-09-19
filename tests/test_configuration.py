import numpy as np
import pytest
from numpy.typing import ArrayLike

from nkscape.configuration import check_configuration


@pytest.mark.parametrize(
    "config",
    [[1, 0, 1], (1, 0, 1), np.array([1, 0, 1]), [True, False, True], [1.0, 0.0, 1.0]],
)
def test_anything_holding_n_zeros_and_ones_is_accepted(config: ArrayLike) -> None:
    row = check_configuration(config, 3)
    assert np.array_equal(row, [1, 0, 1])
    assert row.dtype == np.int_


@pytest.mark.parametrize("config", [[1, 0], [1, 0, 1, 1], [[1, 0, 1]], 1])
def test_wrong_length_or_nesting_is_rejected(config: ArrayLike) -> None:
    with pytest.raises(ValueError, match="must have 3 values"):
        check_configuration(config, 3)


@pytest.mark.parametrize("config", [[1, 2, 1], [-1, 0, 1], [1, 0.5, 1]])
def test_values_other_than_0_and_1_are_rejected(config: ArrayLike) -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        check_configuration(config, 3)


def test_the_input_is_not_changed_or_shared() -> None:
    given = np.array([1, 0, 1])
    row = check_configuration(given, 3)
    row[0] = 0
    assert given[0] == 1
