"""
Reproducibility guards: what a seed produces must never change by accident.

If a test in this file fails, a change has altered the results every user gets for a given seed
(the order random numbers are drawn in, how streams are spawned, how fitness is summed - or a
NumPy upgrade changed one of these). That is never a harmless refactoring. Either undo the
change, or - if it is intended - update the pinned values here and record it in CHANGELOG.md
as a change of results.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

import nkscape as nk


class OneRandomTry(nk.Agent):
    def step(
        self,
        config: NDArray[np.int8],
        landscape: nk.Landscape,
        rng: np.random.Generator,
        population: nk.Population,
    ) -> ArrayLike:
        candidate: NDArray[np.int_] = landscape.neighbours(config)[rng.integers(landscape.n)]
        return candidate if landscape.fitness(candidate) > landscape.fitness(config) else config


def test_pinned_random_k_matrix_for_seed_42() -> None:
    m = nk.InteractionMatrix.random_k(6, 2, rng=42)
    assert m.grid.tolist() == [
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 1],
    ]


def test_pinned_contribution_tables_for_seed_42() -> None:
    t = nk.ContributionTables.random(nk.InteractionMatrix.next_k(3, 1), rng=42)
    assert [table.tolist() for table in t.tables] == [
        [[0.7739560485559633, 0.4388784397520523], [0.8585979199113825, 0.6973680290593639]],
        [[0.09417734788764953, 0.9756223516367559], [0.761139701990353, 0.7860643052769538]],
        [[0.12811363267554587, 0.45038593789556713], [0.37079802423258124, 0.9267649888486018]],
    ]


def test_pinned_random_k_landscape_for_seed_42() -> None:
    land = nk.Landscape.random_k(10, 3, rng=42)
    assert land.matrix.grid[0].tolist() == [1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    assert land.matrix.grid[9].tolist() == [1, 0, 0, 0, 0, 1, 0, 1, 0, 1]
    assert land.tables.tables[0].flat[:3].tolist() == [
        0.4674907799518424,
        0.04644889644868733,
        0.5955100095961371,
    ]
    assert land.fitness([0] * 10) == 0.7054775687210503
    assert land.fitness([1] * 10) == 0.5142926973462552
    assert land.count_peaks() == 15
    assert land.global_max() == 0.785100747115253
    assert land.fitness([1, 1, 1, 1, 1, 0, 1, 0, 1, 1]) == land.global_max()


def test_pinned_next_k_landscape_for_seed_42() -> None:
    land = nk.Landscape.next_k(10, 3, rng=42)
    assert land.fitness([0] * 10) == 0.48805551614964526
    assert land.count_peaks() == 14


def test_pinned_float32_landscape_for_seed_42() -> None:
    land = nk.Landscape.random_k(10, 3, rng=42, dtype=np.float32)
    assert land.fitness([0] * 10) == 0.7054775357246399
    assert land.count_peaks() == 15
    assert land.global_max() == 0.7851007580757141


def test_pinned_random_configuration_for_seed_5() -> None:
    land = nk.Landscape.next_k(10, 0, rng=1)
    assert land.random_configuration(rng=5).tolist() == [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]


def test_entry_p_of_table_i_is_a_fixed_draw_of_the_tables_stream() -> None:
    m = nk.InteractionMatrix.random_k(6, 2, rng=1)
    t = nk.ContributionTables.random(m, rng=42)
    sizes = [table.size for table in t.tables]
    for i in range(m.n):
        for p in range(sizes[i]):
            stream = np.random.Generator(np.random.PCG64(42).advance(sum(sizes[:i]) + p))
            assert stream.random() == t.tables[i].flat[p]


def test_a_landscape_draws_its_tables_from_the_second_spawned_stream() -> None:
    land = nk.Landscape.random_k(6, 2, rng=42)
    sizes = [table.size for table in land.tables.tables]
    for i, p in [(0, 0), (2, 5), (5, 7)]:
        tables_seed = np.random.SeedSequence(42).spawn(2)[1]
        stream = np.random.Generator(np.random.PCG64(tables_seed).advance(sum(sizes[:i]) + p))
        assert stream.random() == land.tables.tables[i].flat[p]


def test_float32_tables_are_the_float64_draws_rounded() -> None:
    m = nk.InteractionMatrix.next_k(8, 3)
    full = nk.ContributionTables.random(m, rng=42)
    half = nk.ContributionTables.random(m, rng=42, dtype=np.float32)
    for a, b in zip(full.tables, half.tables, strict=True):
        assert np.array_equal(a.astype(np.float32), b)


def test_pinned_walks_for_seed_42() -> None:
    land = nk.Landscape.random_k(10, 3, rng=42)
    result = nk.run(land, [OneRandomTry(), OneRandomTry()], periods=12, rng=42)
    walks = [["".join(map(str, config)) for config in agent] for agent in result.configs]
    assert walks[0] == [
        "0011000111",
        "0011000111",
        "0011000111",
        "0011100111",
        "0011100111",
        "0011100111",
        "0011100111",
        "0011100111",
        "0111100111",
        "0111100111",
        "0111100111",
        "0111100111",
        "0111100111",
    ]
    assert walks[1] == ["0001100101"] * 11 + ["0000100101"] * 2
    assert result.fitness[:, -1].tolist() == [0.5389108750872633, 0.6054437480998407]


def test_pinned_hill_climber_walks_for_seed_42() -> None:
    land = nk.Landscape.random_k(10, 3, rng=42)
    agents = [
        nk.agents.HillClimber("first"),
        nk.agents.HillClimber("random"),
        nk.agents.HillClimber("first", long_jumps=True),
        nk.agents.HillClimber("best"),
    ]
    result = nk.run(land, agents, periods=12, rng=42)
    walks = [["".join(map(str, config)) for config in agent] for agent in result.configs]
    assert (
        walks[0]
        == [
            "0011000111",
            "1011000111",
            "1111000111",
            "1111001111",
            "0111001111",
            "0011001111",
            "0011001101",
        ]
        + ["0010001101"] * 6
    )
    assert (
        walks[1]
        == [
            "0001100101",
            "0000100101",
            "0000101101",
            "0000001101",
        ]
        + ["0010001101"] * 9
    )
    assert walks[2] == ["1100001001", "1000001001"] + ["0000000000"] * 11
    assert (
        walks[3]
        == [
            "1100111111",
            "1100111011",
            "1110111011",
            "1111111011",
        ]
        + ["1111101011"] * 9
    )
    assert result.fitness[:, -1].tolist() == [
        0.7092154989468789,
        0.7092154989468789,
        0.7054775687210503,
        0.785100747115253,
    ]
