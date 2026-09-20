from collections import Counter
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from numpy.typing import ArrayLike

import nkscape as nk

Pick = Literal["first", "random", "best"]
PICKS: list[Pick] = ["first", "random", "best"]

LAND = nk.Landscape.random_k(10, 3, rng=1)


def hand_made(n: int, k: int, tables: list[ArrayLike]) -> nk.Landscape:
    matrix = nk.InteractionMatrix.next_k(n, k)
    return nk.Landscape(matrix, nk.ContributionTables(matrix, tables))


def where_they_went(
    land: nk.Landscape, make: Callable[[], nk.Agent], start: list[int]
) -> Counter[str]:
    result = nk.run(land, [make() for _ in range(2000)], periods=1, rng=1, starts=[start] * 2000)
    return Counter("".join(map(str, config)) for config in result.configs[:, 1])


# N = 3, K = 0. From 000 (1/3): 100 is better (1.25/3), 010 worse (0.75/3), 001 best (1.5/3).
ONE_GOOD_ONE_BEST = hand_made(3, 0, [[0.25, 0.5], [0.5, 0.25], [0.25, 0.75]])


@pytest.mark.parametrize("pick", PICKS)
def test_without_interactions_one_wrong_decision_is_fixed_per_period(pick: Pick) -> None:
    land = nk.Landscape.next_k(10, 0, rng=1)
    optimum = land.peaks()[0]
    result = nk.run(land, [nk.agents.HillClimber(pick) for _ in range(20)], periods=12, rng=2)
    wrong = (result.configs != optimum).sum(axis=2)
    for t in range(13):
        assert wrong[:, t].tolist() == np.maximum(wrong[:, 0] - t, 0).tolist()


@pytest.mark.parametrize("pick", PICKS)
def test_a_walk_only_goes_up_one_decision_at_a_time_and_ends_on_a_peak(pick: Pick) -> None:
    result = nk.run(LAND, [nk.agents.HillClimber(pick) for _ in range(30)], periods=40, rng=2)
    changed = (result.configs[:, 1:] != result.configs[:, :-1]).sum(axis=2)
    assert changed.max() == 1
    assert (np.diff(result.fitness, axis=1)[changed == 1] > 0).all()
    assert all(LAND.is_peak(config) for config in result.configs[:, -1])


def test_first_always_takes_the_lowest_numbered_better_decision() -> None:
    went = where_they_went(ONE_GOOD_ONE_BEST, lambda: nk.agents.HillClimber("first"), [0, 0, 0])
    assert went == {"100": 2000}


def test_first_uses_no_randomness() -> None:
    starts = [LAND.random_configuration(rng) for rng in range(20)]
    walks = [
        nk.run(LAND, [nk.agents.HillClimber("first") for _ in range(20)], 15, rng, starts=starts)
        for rng in (1, 2)
    ]
    assert np.array_equal(walks[0].configs, walks[1].configs)


def test_random_takes_every_better_neighbour_equally_often_and_never_a_worse_one() -> None:
    went = where_they_went(ONE_GOOD_ONE_BEST, lambda: nk.agents.HillClimber("random"), [0, 0, 0])
    assert set(went) == {"100", "001"}
    assert 900 < went["100"] < 1100


def test_best_always_takes_the_highest_neighbour() -> None:
    went = where_they_went(ONE_GOOD_ONE_BEST, lambda: nk.agents.HillClimber("best"), [0, 0, 0])
    assert went == {"001": 2000}


def test_best_breaks_an_exact_tie_at_random() -> None:
    land = hand_made(3, 0, [[0.25, 0.75], [0.5, 0.25], [0.25, 0.75]])
    assert land.fitness([1, 0, 0]) == land.fitness([0, 0, 1])
    went = where_they_went(land, lambda: nk.agents.HillClimber("best"), [0, 0, 0])
    assert set(went) == {"100", "001"}
    assert 900 < went["100"] < 1100


def test_a_long_jump_is_taken_when_it_beats_the_neighbour_picked() -> None:
    fitness = [[0.125, 0.5], [0.5, 0.875]]
    land = hand_made(2, 1, [fitness, fitness])
    went = where_they_went(land, lambda: nk.agents.HillClimber("random", long_jumps=True), [0, 0])
    assert set(went) == {"01", "10", "11"}
    assert 400 < went["11"] < 600


def test_a_long_jump_that_is_better_but_not_as_good_as_the_neighbour_is_not_taken() -> None:
    fitness = [[0.125, 0.0625], [0.875, 0.5]]
    land = hand_made(2, 1, [fitness, fitness])
    went = where_they_went(land, lambda: nk.agents.HillClimber("first", long_jumps=True), [0, 0])
    assert went == {"10": 2000}


def test_only_long_jumps_get_an_agent_off_a_peak() -> None:
    peaks = LAND.peaks()
    lowest = peaks[np.argmin([LAND.fitness(peak) for peak in peaks])]
    stuck = nk.run(LAND, [nk.agents.HillClimber("first")], 200, rng=3, starts=[lowest])
    freed = nk.run(
        LAND, [nk.agents.HillClimber("first", long_jumps=True)], 200, rng=3, starts=[lowest]
    )
    assert (stuck.configs[0] == lowest).all()
    assert freed.fitness[0, -1] > freed.fitness[0, 0]
    assert (np.diff(freed.fitness[0]) >= 0).all()
    assert (freed.configs[0, 1:] != freed.configs[0, :-1]).sum(axis=1).max() > 1


@pytest.mark.parametrize("pick", PICKS)
def test_the_walk_is_the_same_with_and_without_the_fitness_cube(pick: Pick) -> None:
    def walk(build_cube: bool) -> nk.RunResult:
        land = nk.Landscape.random_k(10, 3, rng=1)
        agents = [nk.agents.HillClimber(pick, long_jumps=True) for _ in range(10)]
        return nk.run(land, agents, periods=30, rng=4, build_cube=build_cube)

    assert np.array_equal(walk(True).configs, walk(False).configs)
    assert np.array_equal(walk(True).fitness, walk(False).fitness)


def test_pick_must_be_first_or_best() -> None:
    with pytest.raises(ValueError, match='pick must be "first", "random" or "best"'):
        nk.agents.HillClimber("bset")  # type: ignore[arg-type]
