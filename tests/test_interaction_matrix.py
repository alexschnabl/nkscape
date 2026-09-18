import numpy as np
import pytest

import nkscape as nk

FIRM = [
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
]

UNEVEN = [
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
]


def test_valid_matrix_is_accepted() -> None:
    m = nk.InteractionMatrix(FIRM)
    assert m.n == 3
    assert np.array_equal(m.grid, FIRM)


def test_numpy_array_is_accepted() -> None:
    m = nk.InteractionMatrix(np.array(FIRM))
    assert np.array_equal(m.grid, FIRM)


def test_rows_may_have_different_k() -> None:
    m = nk.InteractionMatrix(UNEVEN)
    assert m.n == 4


def test_not_square_is_rejected() -> None:
    with pytest.raises(ValueError, match="square"):
        nk.InteractionMatrix([[1, 1, 0], [0, 1, 1]])


def test_single_row_is_rejected() -> None:
    with pytest.raises(ValueError, match="square"):
        nk.InteractionMatrix([1, 0, 1])  # type: ignore[list-item]


def test_rows_of_different_length_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        nk.InteractionMatrix([[1, 0], [1]])


def test_values_other_than_0_and_1_are_rejected() -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        nk.InteractionMatrix([[1, 2], [0, 1]])


def test_hole_in_diagonal_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"rows \[1\]"):
        nk.InteractionMatrix([[1, 0, 0], [0, 0, 1], [0, 0, 1]])


def test_k_of_uniform_matrix() -> None:
    m = nk.InteractionMatrix(FIRM)
    assert m.k == 1
    assert np.array_equal(m.k_per_row, [1, 1, 1])


def test_k_is_zero_when_only_the_diagonal_is_set() -> None:
    m = nk.InteractionMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert m.k == 0
    assert np.array_equal(m.k_per_row, [0, 0, 0])


def test_k_is_n_minus_1_when_matrix_is_full() -> None:
    m = nk.InteractionMatrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert m.k == 2
    assert np.array_equal(m.k_per_row, [2, 2, 2])


def test_k_per_row_of_uneven_matrix() -> None:
    m = nk.InteractionMatrix(UNEVEN)
    assert np.array_equal(m.k_per_row, [1, 1, 1, 2])


def test_k_is_refused_for_uneven_matrix() -> None:
    m = nk.InteractionMatrix(UNEVEN)
    with pytest.raises(ValueError, match="different K"):
        _ = m.k


def test_k_is_a_plain_int() -> None:
    m = nk.InteractionMatrix(FIRM)
    assert type(m.k) is int
