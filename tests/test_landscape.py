from collections.abc import Callable

import numpy as np
import pytest

import nkscape as nk

Builder = Callable[..., nk.Landscape]

FIRM = nk.InteractionMatrix.next_k(3, 1)
FIRM_TABLES = nk.ContributionTables(
    FIRM,
    [
        [[0.8, 0.1], [0.2, 0.9]],
        [[0.7, 0.2], [0.3, 0.8]],
        [[0.6, 0.2], [0.1, 0.9]],
    ],
)

CUBE = {
    (0, 0, 0): 2.1 / 3,
    (0, 0, 1): 1.2 / 3,
    (0, 1, 0): 1.0 / 3,
    (0, 1, 1): 1.1 / 3,
    (1, 0, 0): 1.0 / 3,
    (1, 0, 1): 1.3 / 3,
    (1, 1, 0): 1.3 / 3,
    (1, 1, 1): 2.6 / 3,
}


def same_tables(a: nk.Landscape, b: nk.Landscape) -> bool:
    pairs = zip(a.tables.tables, b.tables.tables, strict=True)
    return all(np.array_equal(x, y) for x, y in pairs)


def test_fitness_of_the_firm_reproduces_the_cube_from_lesson_1() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    assert land.n == 3
    for config, fitness in CUBE.items():
        assert land.fitness(config) == pytest.approx(fitness)


def test_fitness_is_a_plain_float() -> None:
    assert type(nk.Landscape(FIRM, FIRM_TABLES).fitness([1, 0, 1])) is float


def test_contributions_are_passed_through_from_the_tables() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    assert np.array_equal(land.contributions([1, 0, 1]), FIRM_TABLES.contributions([1, 0, 1]))


def test_tables_built_for_a_different_matrix_are_rejected() -> None:
    identical_looking = nk.InteractionMatrix.next_k(3, 1)
    with pytest.raises(ValueError, match="different interaction matrix"):
        nk.Landscape(identical_looking, FIRM_TABLES)


def test_bad_configuration_is_rejected() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    with pytest.raises(ValueError, match="must have 3 values"):
        land.fitness([1, 0])


def test_with_random_tables_keeps_the_given_matrix() -> None:
    land = nk.Landscape.with_random_tables(FIRM, rng=42)
    assert land.matrix is FIRM
    assert land.tables.matrix is FIRM


def test_with_random_tables_same_seed_gives_same_tables() -> None:
    a = nk.Landscape.with_random_tables(FIRM, rng=42)
    b = nk.Landscape.with_random_tables(FIRM, rng=42)
    c = nk.Landscape.with_random_tables(FIRM, rng=43)
    assert same_tables(a, b)
    assert not same_tables(a, c)


def test_next_k_is_a_shortcut_for_the_long_form() -> None:
    short = nk.Landscape.next_k(10, 3, rng=42)
    long = nk.Landscape.with_random_tables(nk.InteractionMatrix.next_k(10, 3), rng=42)
    assert np.array_equal(short.matrix.grid, long.matrix.grid)
    assert same_tables(short, long)


def test_random_k_same_seed_gives_same_landscape() -> None:
    a = nk.Landscape.random_k(10, 3, rng=42)
    b = nk.Landscape.random_k(10, 3, rng=42)
    assert np.array_equal(a.matrix.grid, b.matrix.grid)
    assert same_tables(a, b)


def test_random_k_different_seeds_give_different_landscapes() -> None:
    a = nk.Landscape.random_k(10, 3, rng=42)
    b = nk.Landscape.random_k(10, 3, rng=43)
    assert not np.array_equal(a.matrix.grid, b.matrix.grid)
    assert not same_tables(a, b)


def test_random_k_uses_separate_streams_for_matrix_and_tables() -> None:
    land = nk.Landscape.random_k(10, 3, rng=42)

    matrix_rng, tables_rng = np.random.default_rng(42).spawn(2)
    expected_matrix = nk.InteractionMatrix.random_k(10, 3, matrix_rng)
    expected = nk.Landscape.with_random_tables(expected_matrix, tables_rng)
    assert np.array_equal(land.matrix.grid, expected.matrix.grid)
    assert same_tables(land, expected)

    shared_bookmark = nk.Landscape.with_random_tables(land.matrix, rng=42)
    assert not same_tables(land, shared_bookmark)


@pytest.mark.parametrize("build", [nk.Landscape.random_k, nk.Landscape.next_k])
@pytest.mark.parametrize(("n", "k"), [(1, 0), (6, 0), (6, 2), (6, 5)])
def test_builders_give_the_requested_n_and_k(build: Builder, n: int, k: int) -> None:
    land = build(n, k, rng=7)
    assert land.n == n
    assert land.matrix.k == k
    assert 0 <= land.fitness([0] * n) < 1


@pytest.mark.parametrize("build", [nk.Landscape.random_k, nk.Landscape.next_k])
@pytest.mark.parametrize(("n", "k"), [(3, -1), (3, 3), (0, 0)])
def test_builders_reject_impossible_n_and_k(build: Builder, n: int, k: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        build(n, k, rng=1)
