import numpy as np
import pytest
from numpy.typing import ArrayLike

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
