import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import nkscape as nk

FIRM_MATRIX = nk.InteractionMatrix.next_k(3, 1)
FIRM = nk.Landscape(
    FIRM_MATRIX,
    nk.ContributionTables(
        FIRM_MATRIX,
        [
            [[0.8, 0.1], [0.2, 0.9]],
            [[0.7, 0.2], [0.3, 0.8]],
            [[0.6, 0.2], [0.1, 0.9]],
        ],
    ),
)


class BestNeighbour(nk.Agent):
    def step(
        self, config: NDArray[np.int_], landscape: nk.Landscape, rng: np.random.Generator
    ) -> ArrayLike:
        options = landscape.neighbour_fitness(config)
        j = int(np.argmax(options))
        best: NDArray[np.int_] = landscape.neighbours(config)[j]
        return best if options[j] > landscape.fitness(config) else config


class OneRandomTry(nk.Agent):
    def step(
        self, config: NDArray[np.int_], landscape: nk.Landscape, rng: np.random.Generator
    ) -> ArrayLike:
        candidate: NDArray[np.int_] = landscape.neighbours(config)[rng.integers(landscape.n)]
        return candidate if landscape.fitness(candidate) > landscape.fitness(config) else config


class CountsItsPeriods(OneRandomTry):
    def __init__(self) -> None:
        self.periods_seen = 0

    def step(
        self, config: NDArray[np.int_], landscape: nk.Landscape, rng: np.random.Generator
    ) -> ArrayLike:
        self.periods_seen += 1
        return super().step(config, landscape, rng)

    def record(self) -> dict[str, float]:
        return {"periods_seen": self.periods_seen, "twice": 2 * self.periods_seen}


class EditsItsInput(nk.Agent):
    def step(
        self, config: NDArray[np.int_], landscape: nk.Landscape, rng: np.random.Generator
    ) -> ArrayLike:
        config[0] = 1 - config[0]
        return config


class Returns(nk.Agent):
    def __init__(self, answer: ArrayLike) -> None:
        self.answer = answer

    def step(
        self, config: NDArray[np.int_], landscape: nk.Landscape, rng: np.random.Generator
    ) -> ArrayLike:
        return self.answer


class ChangesWhatItRecords(OneRandomTry):
    def __init__(self) -> None:
        self.calls = 0

    def record(self) -> dict[str, float]:
        self.calls += 1
        return {"a": 1.0} if self.calls < 3 else {"b": 1.0}


def test_best_neighbour_on_the_firm_gets_stuck_on_the_lower_peak() -> None:
    result = nk.run(FIRM, BestNeighbour(), periods=3, rng=1, start=[0, 1, 0])
    assert result.configs.tolist() == [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert result.fitness == pytest.approx([1.0 / 3, 2.1 / 3, 2.1 / 3, 2.1 / 3])
    assert result.extra == {}


def test_zero_periods_records_only_the_start() -> None:
    result = nk.run(FIRM, BestNeighbour(), periods=0, rng=1, start=[0, 1, 0])
    assert result.configs.tolist() == [[0, 1, 0]]
    assert result.fitness.shape == (1,)


def test_negative_periods_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        nk.run(FIRM, BestNeighbour(), periods=-1, rng=1)


def test_same_seed_gives_the_same_run() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    a = nk.run(land, OneRandomTry(), periods=30, rng=7)
    b = nk.run(land, OneRandomTry(), periods=30, rng=7)
    c = nk.run(land, OneRandomTry(), periods=30, rng=8)
    assert np.array_equal(a.configs, b.configs)
    assert np.array_equal(a.fitness, b.fitness)
    assert not np.array_equal(a.configs, c.configs)


def test_giving_the_start_does_not_change_the_agents_randomness() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    drawn = nk.run(land, OneRandomTry(), periods=30, rng=7)
    given = nk.run(land, OneRandomTry(), periods=30, rng=7, start=drawn.configs[0])
    assert np.array_equal(drawn.configs, given.configs)


def test_a_generator_can_be_passed_and_continues() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    streams = np.random.default_rng(42).spawn(2)
    a = nk.run(land, OneRandomTry(), periods=30, rng=streams[0])
    b = nk.run(land, OneRandomTry(), periods=30, rng=streams[1])
    assert not np.array_equal(a.configs, b.configs)

    replay = np.random.default_rng(42).spawn(2)
    assert np.array_equal(nk.run(land, OneRandomTry(), 30, replay[0]).configs, a.configs)


def test_recorded_fitness_is_the_fitness_of_the_recorded_configuration() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    result = nk.run(land, OneRandomTry(), periods=30, rng=7)
    assert result.configs.shape == (31, 10)
    for config, fitness in zip(result.configs, result.fitness, strict=True):
        assert land.fitness(config) == fitness


def test_fitness_never_falls_for_an_agent_that_only_moves_uphill() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    result = nk.run(land, OneRandomTry(), periods=60, rng=9)
    assert (np.diff(result.fitness) >= 0).all()


def test_an_agent_keeps_its_memory_and_can_record_it() -> None:
    agent = CountsItsPeriods()
    result = nk.run(FIRM, agent, periods=5, rng=1)
    assert agent.periods_seen == 5
    assert result.extra["periods_seen"].tolist() == [0, 1, 2, 3, 4, 5]
    assert result.extra["twice"].tolist() == [0, 2, 4, 6, 8, 10]


def test_changing_the_recorded_names_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"same names in every period.*\['a'\].*\['b'\]"):
        nk.run(FIRM, ChangesWhatItRecords(), periods=5, rng=1)


def test_an_agent_that_edits_its_input_is_stopped() -> None:
    with pytest.raises(ValueError, match="read-only"):
        nk.run(FIRM, EditsItsInput(), periods=2, rng=1)


@pytest.mark.parametrize("answer", [[1, 0], [1, 2, 0], [[1, 0, 1]]])
def test_an_agent_that_returns_no_valid_configuration_is_stopped(answer: ArrayLike) -> None:
    with pytest.raises(ValueError, match="configuration"):
        nk.run(FIRM, Returns(answer), periods=2, rng=1)


def test_agent_cannot_be_created_without_a_step() -> None:
    class Forgot(nk.Agent):
        pass

    with pytest.raises(TypeError, match="step"):
        nk.Agent()  # type: ignore[abstract]
    with pytest.raises(TypeError, match="step"):
        Forgot()  # type: ignore[abstract]


def test_results_are_read_only() -> None:
    result = nk.run(FIRM, CountsItsPeriods(), periods=2, rng=1)
    for array in (result.configs, result.fitness, result.extra["twice"]):
        assert not array.flags.writeable


def test_the_cube_is_built_up_to_n_20_by_default() -> None:
    small = nk.Landscape.next_k(20, 2, rng=1)
    large = nk.Landscape.next_k(21, 2, rng=1)
    nk.run(small, OneRandomTry(), periods=1, rng=1)
    nk.run(large, OneRandomTry(), periods=1, rng=1)
    assert small._fitness_cube is not None
    assert large._fitness_cube is None


def test_build_cube_can_be_forced_or_forbidden() -> None:
    small = nk.Landscape.next_k(8, 2, rng=1)
    large = nk.Landscape.next_k(21, 2, rng=1)
    nk.run(small, OneRandomTry(), periods=1, rng=1, build_cube=False)
    nk.run(large, OneRandomTry(), periods=1, rng=1, build_cube=True)
    assert small._fitness_cube is None
    assert large._fitness_cube is not None


def test_forcing_the_cube_above_the_enumeration_limit_is_refused() -> None:
    land = nk.Landscape.next_k(25, 2, rng=1)
    with pytest.raises(ValueError, match="above the limit of 24"):
        nk.run(land, OneRandomTry(), periods=1, rng=1, build_cube=True)


def test_a_run_gives_the_same_walk_with_and_without_the_cube() -> None:
    with_cube = nk.run(nk.Landscape.random_k(10, 3, rng=1), OneRandomTry(), 40, rng=5)
    without = nk.run(
        nk.Landscape.random_k(10, 3, rng=1), OneRandomTry(), 40, rng=5, build_cube=False
    )
    assert np.array_equal(with_cube.configs, without.configs)
    assert np.array_equal(with_cube.fitness, without.fitness)


def test_a_float32_landscape_records_float32_fitness() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1, dtype=np.float32)
    assert nk.run(land, OneRandomTry(), periods=5, rng=1).fitness.dtype == np.float32
