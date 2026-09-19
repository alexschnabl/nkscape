import tracemalloc
from itertools import product

import numpy as np
import pytest
from numpy.typing import ArrayLike, DTypeLike

import nkscape as nk

FIRM = nk.InteractionMatrix.next_k(3, 1)

# Axes follow depends_on: table 0 is [d0][d1], table 1 is [d1][d2], table 2 is [d0][d2].
FIRM_TABLES = [
    [[0.8, 0.1], [0.2, 0.9]],
    [[0.7, 0.2], [0.3, 0.8]],
    [[0.6, 0.2], [0.1, 0.9]],
]

UNEVEN = nk.InteractionMatrix(
    [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ]
)


def test_hand_made_tables_are_accepted() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    assert t.n == 3
    assert t.matrix is FIRM
    assert len(t.tables) == 3
    for stored, given in zip(t.tables, FIRM_TABLES, strict=True):
        assert np.array_equal(stored, given)


def test_wrong_number_of_tables_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 3 tables"):
        nk.ContributionTables(FIRM, FIRM_TABLES[:2])


def test_too_many_tables_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"expected 3 tables.*got more"):
        nk.ContributionTables(FIRM, [*FIRM_TABLES, FIRM_TABLES[0]])


def test_tables_can_come_from_a_generator() -> None:
    t = nk.ContributionTables(FIRM, (table for table in FIRM_TABLES))
    assert np.array_equal(t.contributions([1, 0, 1]), [0.2, 0.2, 0.9])

    with pytest.raises(ValueError, match=r"expected 3 tables.*got 2"):
        nk.ContributionTables(FIRM, (table for table in FIRM_TABLES[:2]))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_creating_random_tables_needs_little_more_memory_than_the_tables(
    dtype: DTypeLike,
) -> None:
    # K = N - 1: the tables are large. Each float64 draw must be released once it is stored,
    # so the peak stays near the size of the stored tables plus one table in transit.
    m = nk.InteractionMatrix.next_k(16, 15)
    tracemalloc.start()
    t = nk.ContributionTables.random(m, rng=1, dtype=dtype)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stored = sum(table.nbytes for table in t.tables)
    one_draw = 2**16 * 8
    assert peak < stored + 2 * one_draw + 100_000


def test_table_with_wrong_shape_is_rejected() -> None:
    tables: list[ArrayLike] = [[0.3, 0.7], FIRM_TABLES[1], FIRM_TABLES[2]]
    with pytest.raises(ValueError, match=r"table 0 must have shape \(2, 2\).*got shape \(2,\)"):
        nk.ContributionTables(FIRM, tables)


def test_ragged_table_is_rejected() -> None:
    tables = [[[0.8, 0.1], [0.2]], FIRM_TABLES[1], FIRM_TABLES[2]]
    with pytest.raises(ValueError, match="table 0 is not a regular grid"):
        nk.ContributionTables(FIRM, tables)


def test_random_tables_fit_an_uneven_matrix() -> None:
    t = nk.ContributionTables.random(UNEVEN, rng=1)
    assert [table.shape for table in t.tables] == [(2, 2), (2, 2), (2, 2), (2, 2, 2)]


@pytest.mark.parametrize(("n", "k"), [(1, 0), (4, 0), (4, 3), (12, 11)])
def test_random_tables_fit_the_extremes_of_k(n: int, k: int) -> None:
    t = nk.ContributionTables.random(nk.InteractionMatrix.next_k(n, k), rng=1)
    assert t.n == n
    assert all(table.shape == (2,) * (k + 1) for table in t.tables)


def test_random_same_seed_gives_same_tables() -> None:
    a = nk.ContributionTables.random(FIRM, rng=42)
    b = nk.ContributionTables.random(FIRM, rng=42)
    for x, y in zip(a.tables, b.tables, strict=True):
        assert np.array_equal(x, y)


def test_random_different_seeds_give_different_tables() -> None:
    a = nk.ContributionTables.random(FIRM, rng=42)
    b = nk.ContributionTables.random(FIRM, rng=43)
    assert not np.array_equal(a.tables[0], b.tables[0])


def test_random_values_are_between_0_and_1() -> None:
    t = nk.ContributionTables.random(nk.InteractionMatrix.next_k(12, 5), rng=3)
    for table in t.tables:
        assert (table >= 0).all()
        assert (table < 1).all()


def test_random_with_a_generator_continues_and_can_be_replayed() -> None:
    gen = np.random.default_rng(42)
    a = nk.ContributionTables.random(FIRM, rng=gen)
    b = nk.ContributionTables.random(FIRM, rng=gen)
    assert not np.array_equal(a.tables[0], b.tables[0])

    replay = np.random.default_rng(42)
    a2 = nk.ContributionTables.random(FIRM, rng=replay)
    b2 = nk.ContributionTables.random(FIRM, rng=replay)
    assert np.array_equal(a2.tables[0], a.tables[0])
    assert np.array_equal(b2.tables[0], b.tables[0])


def test_contributions_of_the_firm() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    assert np.array_equal(t.contributions([1, 0, 1]), [0.2, 0.2, 0.9])


def test_contributions_reproduce_the_cube_from_lesson_1() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    cube = {
        (0, 0, 0): 2.1 / 3,
        (0, 0, 1): 1.2 / 3,
        (0, 1, 0): 1.0 / 3,
        (0, 1, 1): 1.1 / 3,
        (1, 0, 0): 1.0 / 3,
        (1, 0, 1): 1.3 / 3,
        (1, 1, 0): 1.3 / 3,
        (1, 1, 1): 2.6 / 3,
    }
    for config, fitness in cube.items():
        assert t.contributions(config).mean() == pytest.approx(fitness)


def test_contributions_accept_list_tuple_and_numpy_row() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    expected = [0.2, 0.2, 0.9]
    assert np.array_equal(t.contributions([1, 0, 1]), expected)
    assert np.array_equal(t.contributions((1, 0, 1)), expected)
    assert np.array_equal(t.contributions(np.array([1, 0, 1])), expected)
    assert np.array_equal(t.contributions([True, False, True]), expected)


def test_contributions_are_a_row_of_n_decimals() -> None:
    result = nk.ContributionTables(FIRM, FIRM_TABLES).contributions([0, 0, 0])
    assert result.shape == (3,)
    assert result.dtype == np.float64


def test_configuration_of_wrong_length_is_rejected() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    with pytest.raises(ValueError, match=r"must have 3 values.*got shape \(2,\)"):
        t.contributions([1, 0])


def test_nested_configuration_is_rejected() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    with pytest.raises(ValueError, match=r"got shape \(1, 3\)"):
        t.contributions([[1, 0, 1]])


def test_configuration_with_other_values_is_rejected() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    with pytest.raises(ValueError, match="0 and 1"):
        t.contributions([1, 2, 1])


def test_contributions_of_an_uneven_matrix() -> None:
    t = nk.ContributionTables.random(UNEVEN, rng=5)
    config = [1, 0, 1, 1]
    result = t.contributions(config)
    assert result[0] == t.tables[0][1, 0]
    assert result[3] == t.tables[3][0, 1, 1]


def test_with_k_zero_each_contribution_depends_only_on_its_own_decision() -> None:
    t = nk.ContributionTables.random(nk.InteractionMatrix.next_k(4, 0), rng=5)
    result = t.contributions([1, 0, 0, 1])
    assert np.array_equal(result, [t.tables[0][1], t.tables[1][0], t.tables[2][0], t.tables[3][1]])


def test_flipping_a_decision_changes_exactly_the_contributions_it_affects() -> None:
    m = nk.InteractionMatrix.random_k(6, 2, rng=11)
    t = nk.ContributionTables.random(m, rng=12)
    config = np.array([1, 0, 0, 1, 1, 0])
    before = t.contributions(config)
    for j in range(m.n):
        flipped = config.copy()
        flipped[j] = 1 - flipped[j]
        changed = np.flatnonzero(t.contributions(flipped) != before)
        assert np.array_equal(changed, m.affects(j))


def test_contribution_cube_of_the_firm() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    cube = t.contribution_cube(2)
    assert cube.shape == (2, 2, 2)
    assert cube[1, 0, 1] == 0.9
    assert cube[1, 1, 1] == 0.9
    assert cube[0, 0, 1] == 0.2


def test_contribution_cube_agrees_with_contributions_everywhere() -> None:
    t = nk.ContributionTables.random(UNEVEN, rng=8)
    for config in product([0, 1], repeat=t.n):
        expected = t.contributions(config)
        for i in range(t.n):
            assert t.contribution_cube(i)[config] == expected[i]


def test_contribution_cube_copies_nothing_and_is_read_only() -> None:
    t = nk.ContributionTables.random(UNEVEN, rng=8)
    cube = t.contribution_cube(0)
    assert np.shares_memory(cube, t.tables[0])
    assert not cube.flags.writeable


def test_tables_are_float64_by_default() -> None:
    assert nk.ContributionTables(FIRM, FIRM_TABLES).dtype == np.float64
    t = nk.ContributionTables.random(FIRM, rng=1)
    assert t.dtype == np.float64
    assert all(table.dtype == np.float64 for table in t.tables)


@pytest.mark.parametrize("dtype", [np.float32, "float32", np.dtype(np.float32)])
def test_float32_tables_use_half_the_memory(dtype: DTypeLike) -> None:
    m = nk.InteractionMatrix.next_k(10, 5)
    full = nk.ContributionTables.random(m, rng=1)
    half = nk.ContributionTables.random(m, rng=1, dtype=dtype)
    assert half.dtype == np.float32
    assert all(table.dtype == np.float32 for table in half.tables)
    assert sum(t.nbytes for t in half.tables) * 2 == sum(t.nbytes for t in full.tables)
    assert half.contributions([0] * 10).dtype == np.float32
    assert half.contribution_cube(0).dtype == np.float32


def test_hand_made_tables_can_be_float32() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES, dtype=np.float32)
    assert t.contributions([1, 0, 1]) == pytest.approx([0.2, 0.2, 0.9], rel=1e-6)


@pytest.mark.parametrize("dtype", [np.float16, int, "complex128"])
def test_other_dtypes_are_rejected(dtype: DTypeLike) -> None:
    with pytest.raises(ValueError, match="float64 or float32"):
        nk.ContributionTables.random(FIRM, rng=1, dtype=dtype)


def test_writing_into_a_table_is_refused() -> None:
    t = nk.ContributionTables(FIRM, FIRM_TABLES)
    with pytest.raises(ValueError, match="read-only"):
        t.tables[0][1, 0] = 0.5
    assert t.tables[0][1, 0] == 0.2


def test_the_users_own_arrays_stay_writable_and_do_not_leak_in() -> None:
    given = [np.array(table) for table in FIRM_TABLES]
    t = nk.ContributionTables(FIRM, given)
    given[0][1, 0] = 0.5
    assert given[0][1, 0] == 0.5
    assert t.tables[0][1, 0] == 0.2
