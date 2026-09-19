from collections.abc import Callable, Iterator
from itertools import product

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


def flip(config: tuple[int, ...], j: int) -> tuple[int, ...]:
    return (*config[:j], 1 - config[j], *config[j + 1 :])


def test_fitness_cube_of_the_firm() -> None:
    cube = nk.Landscape(FIRM, FIRM_TABLES).fitness_cube()
    assert cube.shape == (2, 2, 2)
    for config, fitness in CUBE.items():
        assert cube[config] == pytest.approx(fitness)


def test_peaks_of_the_firm() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    assert np.array_equal(land.peaks(), [[0, 0, 0], [1, 1, 1]])
    assert land.count_peaks() == 2
    assert land.global_max() == pytest.approx(2.6 / 3)
    assert type(land.global_max()) is float


@pytest.mark.parametrize("k", [0, 2, 7])
def test_fitness_cube_equals_fitness_bit_for_bit(k: int) -> None:
    land = nk.Landscape.random_k(8, k, rng=21)
    cube = land.fitness_cube()
    for config in product([0, 1], repeat=land.n):
        assert cube[config] == land.fitness(config)


@pytest.mark.parametrize("k", [0, 5, 19])
def test_fitness_cube_equals_fitness_bit_for_bit_for_larger_n(k: int) -> None:
    looked_up = nk.Landscape.random_k(20, k, rng=24)
    cube = nk.Landscape.random_k(20, k, rng=24).fitness_cube()
    gen = np.random.default_rng(25)
    for _ in range(300):
        config = looked_up.random_configuration(gen)
        assert cube[tuple(config)] == looked_up.fitness(config)


@pytest.mark.parametrize("k", [0, 3, 9])
def test_count_peaks_is_the_number_of_peaks(k: int) -> None:
    land = nk.Landscape.random_k(10, k, rng=26)
    assert land.count_peaks() == len(land.peaks())
    assert type(land.count_peaks()) is int


@pytest.mark.parametrize("k", [0, 2, 7])
def test_peaks_are_exactly_the_configurations_no_flip_improves(k: int) -> None:
    land = nk.Landscape.random_k(8, k, rng=22)
    peaks = {tuple(int(d) for d in row) for row in land.peaks()}
    for config in product([0, 1], repeat=land.n):
        here = land.fitness(config)
        no_flip_improves = all(land.fitness(flip(config, j)) <= here for j in range(land.n))
        assert (config in peaks) == no_flip_improves


@pytest.mark.parametrize("n", [1, 5, 10])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_k_zero_has_exactly_one_peak(n: int, seed: int) -> None:
    assert nk.Landscape.random_k(n, 0, rng=seed).count_peaks() == 1


def test_the_global_maximum_is_a_peak() -> None:
    land = nk.Landscape.random_k(10, 4, rng=23)
    assert land.global_max() in [land.fitness(peak) for peak in land.peaks()]


def test_a_tie_between_neighbours_counts_as_a_peak() -> None:
    m = nk.InteractionMatrix.next_k(2, 0)
    tables = nk.ContributionTables(m, [[0.5, 0.5], [0.2, 0.8]])
    land = nk.Landscape(m, tables)
    assert np.array_equal(land.peaks(), [[0, 1], [1, 1]])


def test_fitness_cube_is_read_only_and_computed_once() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    cube = land.fitness_cube()
    assert land.fitness_cube() is cube
    with pytest.raises(ValueError, match="read-only"):
        cube[0, 0, 0] = 1.0


def test_looking_at_every_configuration_is_refused_above_the_limit() -> None:
    land = nk.Landscape.next_k(25, 0, rng=1)
    for look in (land.fitness_cube, land.peaks, land.count_peaks, land.global_max):
        with pytest.raises(ValueError, match="above the limit of 24"):
            look()


def test_looking_at_every_configuration_works_at_the_limit() -> None:
    land = nk.Landscape.next_k(24, 0, rng=1)
    assert land.fitness_cube().shape == (2,) * 24


@pytest.fixture
def restore_enumeration_limit() -> Iterator[None]:
    yield
    nk.set_enumeration_limit(24)


def test_the_default_enumeration_limit_is_24() -> None:
    assert nk.get_enumeration_limit() == 24


@pytest.mark.usefixtures("restore_enumeration_limit")
def test_the_enumeration_limit_can_be_changed() -> None:
    nk.set_enumeration_limit(5)
    assert nk.get_enumeration_limit() == 5
    assert nk.Landscape.next_k(5, 1, rng=1).count_peaks() >= 1
    with pytest.raises(ValueError, match="above the limit of 5"):
        nk.Landscape.next_k(6, 1, rng=1).fitness_cube()

    nk.set_enumeration_limit(6)
    assert nk.Landscape.next_k(6, 1, rng=1).count_peaks() >= 1


def test_the_refusal_says_how_much_memory_and_how_to_raise_the_limit() -> None:
    land = nk.Landscape.next_k(30, 0, rng=1)
    with pytest.raises(
        ValueError, match=r"about 10\.7 GB of memory with float64.*nk\.set_enumeration_limit\(30\)"
    ):
        land.fitness_cube()


def test_an_enumeration_limit_below_1_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        nk.set_enumeration_limit(0)
    assert nk.get_enumeration_limit() == 24


def test_neighbours_of_the_firm() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    assert np.array_equal(land.neighbours([0, 1, 0]), [[1, 1, 0], [0, 0, 0], [0, 1, 1]])


def test_row_j_of_neighbours_differs_in_exactly_decision_j() -> None:
    land = nk.Landscape.random_k(12, 3, rng=31)
    config = land.random_configuration(rng=32)
    neighbours = land.neighbours(config)
    assert neighbours.shape == (12, 12)
    for j in range(land.n):
        assert np.array_equal(np.flatnonzero(neighbours[j] != config), [j])


def test_neighbour_fitness_of_the_firm() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    assert land.neighbour_fitness([0, 1, 0]) == pytest.approx([1.3 / 3, 2.1 / 3, 1.1 / 3])


@pytest.mark.parametrize("k", [0, 2, 7])
def test_helpers_give_identical_results_with_and_without_the_cube(k: int) -> None:
    looked_up = nk.Landscape.random_k(8, k, rng=33)
    from_cube = nk.Landscape.random_k(8, k, rng=33)
    from_cube.fitness_cube()
    for config in product([0, 1], repeat=8):
        assert looked_up.fitness(config) == from_cube.fitness(config)
        assert np.array_equal(
            looked_up.neighbour_fitness(config), from_cube.neighbour_fitness(config)
        )
        assert looked_up.is_peak(config) == from_cube.is_peak(config)


def test_the_helpers_never_build_the_cube() -> None:
    land = nk.Landscape.random_k(8, 2, rng=34)
    config = land.random_configuration(rng=35)
    land.fitness(config)
    land.neighbours(config)
    land.neighbour_fitness(config)
    land.is_peak(config)
    assert land._fitness_cube is None


@pytest.mark.parametrize("k", [0, 2, 7])
def test_is_peak_agrees_with_peaks(k: int) -> None:
    land = nk.Landscape.random_k(8, k, rng=36)
    peaks = {tuple(int(d) for d in row) for row in land.peaks()}
    for config in product([0, 1], repeat=8):
        assert land.is_peak(config) == (config in peaks)


def test_is_peak_of_the_firm() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    assert land.is_peak([0, 0, 0])
    assert land.is_peak([1, 1, 1])
    assert not land.is_peak([0, 1, 0])
    assert type(land.is_peak([0, 0, 0])) is bool


def test_the_helpers_work_above_the_enumeration_limit() -> None:
    land = nk.Landscape.next_k(40, 2, rng=37)
    config = land.random_configuration(rng=38)
    assert land.neighbours(config).shape == (40, 40)
    assert land.neighbour_fitness(config).shape == (40,)
    assert land.is_peak(config) in (True, False)


def test_neighbours_can_be_changed_without_harming_the_landscape() -> None:
    land = nk.Landscape(FIRM, FIRM_TABLES)
    neighbours = land.neighbours([0, 1, 0])
    neighbours[0, 0] = 0
    assert np.array_equal(land.neighbours([0, 1, 0]), [[1, 1, 0], [0, 0, 0], [0, 1, 1]])


def test_random_configuration_is_n_zeros_and_ones() -> None:
    land = nk.Landscape.next_k(12, 2, rng=39)
    config = land.random_configuration(rng=40)
    assert config.shape == (12,)
    assert set(config.tolist()) <= {0, 1}
    assert land.fitness(config) >= 0


def test_random_configuration_is_seeded() -> None:
    land = nk.Landscape.next_k(12, 2, rng=39)
    assert np.array_equal(land.random_configuration(rng=1), land.random_configuration(rng=1))
    assert not np.array_equal(land.random_configuration(rng=1), land.random_configuration(rng=2))

    gen = np.random.default_rng(3)
    a, b = land.random_configuration(gen), land.random_configuration(gen)
    assert not np.array_equal(a, b)
    replay = np.random.default_rng(3)
    assert np.array_equal(land.random_configuration(replay), a)
    assert np.array_equal(land.random_configuration(replay), b)


def test_random_configurations_are_about_half_ones() -> None:
    land = nk.Landscape.next_k(20, 0, rng=39)
    gen = np.random.default_rng(4)
    share = np.mean([land.random_configuration(gen) for _ in range(2000)])
    assert share == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("build", [nk.Landscape.random_k, nk.Landscape.next_k])
def test_a_float32_landscape_is_float32_throughout(build: Builder) -> None:
    land = build(10, 3, rng=41, dtype=np.float32)
    config = land.random_configuration(rng=42)
    assert land.tables.dtype == np.float32
    assert land.contributions(config).dtype == np.float32
    assert land.neighbour_fitness(config).dtype == np.float32
    cube = land.fitness_cube()
    assert cube.dtype == np.float32
    assert cube.nbytes == 4 * 2**10
    assert land.neighbour_fitness(config).dtype == np.float32
    assert type(land.fitness(config)) is float


@pytest.mark.parametrize("k", [0, 5, 15])
def test_a_float32_cube_equals_float32_fitness_bit_for_bit(k: int) -> None:
    looked_up = nk.Landscape.random_k(16, k, rng=43, dtype=np.float32)
    from_cube = nk.Landscape.random_k(16, k, rng=43, dtype=np.float32)
    from_cube.fitness_cube()
    gen = np.random.default_rng(44)
    for _ in range(300):
        config = looked_up.random_configuration(gen)
        assert looked_up.fitness(config) == from_cube.fitness(config)
        assert np.array_equal(
            looked_up.neighbour_fitness(config), from_cube.neighbour_fitness(config)
        )


def test_a_float32_landscape_is_the_float64_landscape_rounded() -> None:
    full = nk.Landscape.random_k(10, 3, rng=45)
    half = nk.Landscape.random_k(10, 3, rng=45, dtype=np.float32)
    assert np.array_equal(full.matrix.grid, half.matrix.grid)
    assert np.allclose(full.fitness_cube(), half.fitness_cube(), atol=1e-6)
    assert half.count_peaks() == full.count_peaks()


def test_with_random_tables_passes_the_dtype_on() -> None:
    land = nk.Landscape.with_random_tables(FIRM, rng=46, dtype=np.float32)
    assert land.tables.dtype == np.float32


def test_the_refusal_quotes_the_memory_for_the_landscapes_dtype() -> None:
    land = nk.Landscape.next_k(30, 0, rng=1, dtype=np.float32)
    with pytest.raises(ValueError, match=r"about 6\.4 GB of memory with float32"):
        land.fitness_cube()


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
