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


def test_writing_into_the_grid_is_refused() -> None:
    m = nk.InteractionMatrix(FIRM)
    with pytest.raises(ValueError, match="read-only"):
        m.grid[1, 1] = 0
    assert np.array_equal(m.grid, FIRM)


def test_the_users_own_array_stays_writable() -> None:
    rows = np.array(FIRM)
    nk.InteractionMatrix(rows)
    rows[0, 2] = 1
    assert rows[0, 2] == 1


def test_later_changes_to_the_input_do_not_leak_in() -> None:
    rows = np.array(FIRM)
    m = nk.InteractionMatrix(rows)
    rows[0, 2] = 1
    assert np.array_equal(m.grid, FIRM)


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


def test_depends_on_reads_rows() -> None:
    m = nk.InteractionMatrix(FIRM)
    assert np.array_equal(m.depends_on(0), [0, 1])
    assert np.array_equal(m.depends_on(1), [1, 2])
    assert np.array_equal(m.depends_on(2), [0, 2])


def test_affects_reads_columns() -> None:
    m = nk.InteractionMatrix(FIRM)
    assert np.array_equal(m.affects(0), [0, 2])
    assert np.array_equal(m.affects(1), [0, 1])
    assert np.array_equal(m.affects(2), [1, 2])


def test_depends_on_and_affects_of_uneven_matrix() -> None:
    m = nk.InteractionMatrix(UNEVEN)
    assert np.array_equal(m.depends_on(3), [1, 2, 3])
    assert np.array_equal(m.affects(1), [0, 1, 3])


def test_depends_on_agrees_with_k_per_row() -> None:
    m = nk.InteractionMatrix(UNEVEN)
    for i in range(m.n):
        assert len(m.depends_on(i)) == m.k_per_row[i] + 1


def test_depends_on_and_affects_tell_the_same_story() -> None:
    m = nk.InteractionMatrix(UNEVEN)
    for i in range(m.n):
        for j in range(m.n):
            assert (j in m.depends_on(i)) == (i in m.affects(j))


def test_str_of_the_firm() -> None:
    m = nk.InteractionMatrix(FIRM)
    expected = "\n".join(
        [
            "   d0 d1 d2",
            "c0  X  X  .   K=1",
            "c1  .  X  X   K=1",
            "c2  X  .  X   K=1",
        ]
    )
    assert str(m) == expected


def test_str_shows_k_of_each_row() -> None:
    lines = str(nk.InteractionMatrix(UNEVEN)).splitlines()
    assert len(lines) == 5
    assert [line.endswith("K=1") for line in lines[1:]] == [True, True, True, False]
    assert lines[4].endswith("K=2")


def test_str_stays_aligned_for_two_digit_labels() -> None:
    n, k = 12, 3
    rows = [[1 if (j - i) % n <= k else 0 for j in range(n)] for i in range(n)]
    lines = str(nk.InteractionMatrix(rows)).splitlines()
    assert len(lines) == n + 1
    header = lines[0]
    assert header.endswith("d10 d11")
    for line in lines[1:]:
        assert len(line.split("   K=")[0]) == len(header)


def test_next_k_builds_the_firm() -> None:
    assert np.array_equal(nk.InteractionMatrix.next_k(3, 1).grid, FIRM)


def test_next_k_with_k_zero_is_only_the_diagonal() -> None:
    assert np.array_equal(nk.InteractionMatrix.next_k(4, 0).grid, np.eye(4))


def test_next_k_with_k_n_minus_1_is_all_ones() -> None:
    assert np.array_equal(nk.InteractionMatrix.next_k(4, 3).grid, np.ones((4, 4)))


@pytest.mark.parametrize(("n", "k"), [(1, 0), (3, 1), (6, 2), (12, 3), (12, 11)])
def test_next_k_has_the_requested_k_in_rows_and_columns(n: int, k: int) -> None:
    m = nk.InteractionMatrix.next_k(n, k)
    assert m.n == n
    assert m.k == k
    assert np.array_equal(m.grid.sum(axis=0), [k + 1] * n)


@pytest.mark.parametrize(("n", "k"), [(3, -1), (3, 3), (0, 0)])
def test_next_k_rejects_impossible_n_and_k(n: int, k: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        nk.InteractionMatrix.next_k(n, k)


def test_random_k_same_seed_gives_same_matrix() -> None:
    a = nk.InteractionMatrix.random_k(12, 3, rng=42)
    b = nk.InteractionMatrix.random_k(12, 3, rng=42)
    assert np.array_equal(a.grid, b.grid)


def test_random_k_different_seeds_give_different_matrices() -> None:
    a = nk.InteractionMatrix.random_k(12, 3, rng=42)
    b = nk.InteractionMatrix.random_k(12, 3, rng=43)
    assert not np.array_equal(a.grid, b.grid)


@pytest.mark.parametrize(("n", "k"), [(1, 0), (3, 1), (6, 2), (12, 3), (12, 11)])
def test_random_k_has_the_requested_k_in_every_row(n: int, k: int) -> None:
    m = nk.InteractionMatrix.random_k(n, k, rng=7)
    assert m.n == n
    assert m.k == k


def test_random_k_with_a_generator_continues_and_can_be_replayed() -> None:
    gen = np.random.default_rng(42)
    a = nk.InteractionMatrix.random_k(12, 3, rng=gen)
    b = nk.InteractionMatrix.random_k(12, 3, rng=gen)
    assert not np.array_equal(a.grid, b.grid)

    replay = np.random.default_rng(42)
    assert np.array_equal(nk.InteractionMatrix.random_k(12, 3, rng=replay).grid, a.grid)
    assert np.array_equal(nk.InteractionMatrix.random_k(12, 3, rng=replay).grid, b.grid)


def test_random_k_extremes_leave_nothing_to_chance() -> None:
    assert np.array_equal(nk.InteractionMatrix.random_k(4, 0, rng=1).grid, np.eye(4))
    assert np.array_equal(nk.InteractionMatrix.random_k(4, 3, rng=1).grid, np.ones((4, 4)))


@pytest.mark.parametrize(("n", "k"), [(3, -1), (3, 3), (0, 0)])
def test_random_k_rejects_impossible_n_and_k(n: int, k: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        nk.InteractionMatrix.random_k(n, k, rng=1)


def test_index_out_of_range_is_rejected() -> None:
    m = nk.InteractionMatrix(FIRM)
    with pytest.raises(IndexError):
        m.depends_on(5)
    with pytest.raises(IndexError):
        m.affects(5)
