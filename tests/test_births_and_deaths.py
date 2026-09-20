from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import nkscape as nk

LAND = nk.Landscape.random_k(8, 2, rng=1)
Config = NDArray[np.int8]
Proposals = tuple[NDArray[np.int8], ...]
Rng = np.random.Generator
Plan = dict[int, tuple[list[int], list[ArrayLike]]]


class OneRandomTry(nk.Agent):
    def __init__(self) -> None:
        self.before: list[int] = []
        self.stepped: list[int] = []
        self.after: list[int] = []
        self.seen_alive: list[list[bool]] = []

    def before_period(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> None:
        self.before.append(population.period)

    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        self.stepped.append(population.period)
        self.seen_alive.append(population.alive.tolist())
        candidate: NDArray[np.int_] = landscape.neighbours(config)[rng.integers(landscape.n)]
        return candidate if landscape.fitness(candidate) > landscape.fitness(config) else config

    def after_period(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> None:
        self.after.append(population.period)

    def record(self) -> dict[str, float]:
        return {"steps": len(self.stepped)}


class Scripted(nk.PopulationRule):
    def __init__(self, plan: Plan, make: Callable[[], nk.Agent] = OneRandomTry) -> None:
        self.plan = plan
        self.make = make
        self.newcomers: list[nk.Agent] = []
        self.proposals_seen: list[list[list[list[int]]]] = []
        self.snapshots: list[nk.Population] = []

    def decide(
        self,
        population: nk.Population,
        proposals: Proposals,
        landscape: nk.Landscape,
        rng: Rng,
    ) -> nk.PopulationDecision:
        self.snapshots.append(population)
        self.proposals_seen.append([ranked.tolist() for ranked in proposals])
        dies, starts = self.plan.get(population.period, ([], []))
        born = [(self.make(), start) for start in starts]
        self.newcomers += [agent for agent, _ in born]
        return nk.PopulationDecision([ranked[0] for ranked in proposals], dies=dies, born=born)


ZEROS, ONES = [0] * 8, [1] * 8


def test_without_births_or_deaths_everyone_is_alive_throughout() -> None:
    agents = [OneRandomTry() for _ in range(3)]
    result = nk.run(LAND, agents, periods=4, rng=1)
    assert result.alive.all()
    assert result.alive.shape == (3, 5)
    assert all(a is b for a, b in zip(result.agents, agents, strict=True))


def test_an_agent_that_dies_keeps_its_row_but_is_never_called_again() -> None:
    agents = [OneRandomTry() for _ in range(3)]
    result = nk.run(LAND, agents, periods=4, rng=1, rules=[Scripted({2: ([1], [])})])
    assert result.alive[1].tolist() == [True, True, False, False, False]
    assert result.alive[0].all()
    assert (result.configs[1, 2:] == -1).all()
    assert np.isnan(result.fitness[1, 2:]).all()
    assert not np.isnan(result.fitness[1, :2]).any()
    assert agents[1].before == agents[1].stepped == [1, 2]
    assert agents[1].after == [1]
    assert agents[0].stepped == [1, 2, 3, 4]


def test_the_snapshot_and_the_proposals_mark_who_does_not_exist() -> None:
    rule = Scripted({1: ([1], [])})
    agents = [OneRandomTry() for _ in range(3)]
    nk.run(LAND, agents, periods=3, rng=1, rules=[rule])
    assert agents[0].seen_alive == [[True, True, True], [True, False, True], [True, False, True]]
    later = rule.snapshots[2]
    assert later.alive.tolist() == [True, False, True]
    assert later.configs[1].tolist() == [-1] * 8
    assert np.isnan(later.fitness[1])
    assert rule.proposals_seen[2][1] == [[-1] * 8]


def test_an_agent_that_is_born_gets_a_new_row_and_starts_next_period() -> None:
    rule = Scripted({2: ([], [ONES])})
    result = nk.run(LAND, [OneRandomTry()], periods=4, rng=1, starts=[ZEROS], rules=[rule])
    newcomer = rule.newcomers[0]
    assert len(result.agents) == 2
    assert result.agents[1] is newcomer
    assert result.alive[1].tolist() == [False, False, True, True, True]
    assert (result.configs[1, :2] == -1).all()
    assert result.configs[1, 2].tolist() == ONES
    assert result.fitness[1, 2] == LAND.fitness(ONES)
    assert isinstance(newcomer, OneRandomTry)
    assert newcomer.before == newcomer.stepped == newcomer.after == [3, 4]


def test_record_is_empty_before_birth_and_after_death() -> None:
    rule = Scripted({1: ([], [ONES]), 3: ([0], [])})
    result = nk.run(LAND, [OneRandomTry()], periods=4, rng=1, rules=[rule])
    steps = result.extra["steps"]
    assert steps[0, :3].tolist() == [0, 1, 2]
    assert np.isnan(steps[0, 3:]).all()
    assert np.isnan(steps[1, 0])
    assert steps[1, 1:].tolist() == [0, 1, 2, 3]


def test_the_arrays_grow_as_the_population_does() -> None:
    class EveryoneFoundsOne(nk.PopulationRule):
        def decide(
            self,
            population: nk.Population,
            proposals: Proposals,
            landscape: nk.Landscape,
            rng: Rng,
        ) -> nk.PopulationDecision:
            founders = np.flatnonzero(population.alive)
            born = [(OneRandomTry(), population.configs[a]) for a in founders]
            return nk.PopulationDecision([ranked[0] for ranked in proposals], born=born)

    result = nk.run(LAND, [OneRandomTry(), OneRandomTry()], 5, rng=1, rules=[EveryoneFoundsOne()])
    assert result.alive.sum(axis=0).tolist() == [2, 4, 8, 16, 32, 64]
    assert result.configs.shape == (64, 6, 8)
    assert len(result.agents) == 64
    assert len({id(agent) for agent in result.agents}) == 64
    assert result.extra["steps"].shape == (64, 6)
    for a in range(64):
        born = int(result.alive[a].argmax())
        assert result.extra["steps"][a, born:].tolist() == list(range(6 - born))
    assert np.array_equal(np.isnan(result.fitness), ~result.alive)


def test_every_newcomer_has_its_own_random_stream() -> None:
    rule = Scripted({1: ([], [ZEROS, ZEROS, ZEROS])})
    result = nk.run(LAND, [OneRandomTry()], periods=12, rng=3, rules=[rule])
    walks = {tuple(map(tuple, result.configs[a, 1:])) for a in (1, 2, 3)}
    assert len(walks) == 3


def test_births_and_deaths_from_several_rules_are_combined() -> None:
    rules = [Scripted({1: ([0], [ONES])}), Scripted({1: ([2], [ZEROS, ZEROS])})]
    result = nk.run(LAND, [OneRandomTry() for _ in range(3)], periods=2, rng=1, rules=rules)
    assert result.alive[:, 1].tolist() == [False, True, False, True, True, True]
    assert rules[1].snapshots[0].alive.tolist() == [True, True, True]


def test_the_same_seed_gives_the_same_births_deaths_and_walks() -> None:
    def history() -> nk.RunResult:
        rule = Scripted({1: ([1], [ONES]), 3: ([0], [ZEROS, ONES])})
        return nk.run(LAND, [OneRandomTry() for _ in range(3)], periods=8, rng=5, rules=[rule])

    first, second = history(), history()
    assert np.array_equal(first.alive, second.alive)
    assert np.array_equal(first.configs, second.configs)


def test_a_population_may_die_out_and_be_refounded() -> None:
    rule = Scripted({1: ([0, 1], []), 4: ([], [ONES])})
    result = nk.run(LAND, [OneRandomTry(), OneRandomTry()], periods=5, rng=1, rules=[rule])
    assert result.alive.sum(axis=0).tolist() == [2, 0, 0, 0, 1, 1]
    assert [snapshot.period for snapshot in rule.snapshots] == [1, 2, 3, 4, 5]


def test_a_plain_array_still_means_nobody_dies_and_nobody_is_born() -> None:
    class KeepsFirstChoices(nk.PopulationRule):
        def decide(
            self,
            population: nk.Population,
            proposals: Proposals,
            landscape: nk.Landscape,
            rng: Rng,
        ) -> ArrayLike:
            return [ranked[0] for ranked in proposals]

    with_rule = nk.run(
        LAND, [OneRandomTry() for _ in range(3)], 6, rng=2, rules=[KeepsFirstChoices()]
    )
    without = nk.run(LAND, [OneRandomTry() for _ in range(3)], 6, rng=2)
    assert np.array_equal(with_rule.configs, without.configs)
    assert with_rule.alive.all()


@pytest.mark.parametrize("victim", [1, 7, -1])
def test_letting_an_agent_die_that_does_not_exist_is_rejected(victim: int) -> None:
    rule = Scripted({1: ([1], []), 2: ([victim], [])})
    with pytest.raises(ValueError, match=f"agent {victim} die, which does not exist"):
        nk.run(LAND, [OneRandomTry() for _ in range(3)], periods=3, rng=1, rules=[rule])


def test_a_newcomer_needs_a_valid_configuration() -> None:
    with pytest.raises(ValueError, match="expected 8 values"):
        nk.run(LAND, [OneRandomTry()], periods=2, rng=1, rules=[Scripted({1: ([], [[1, 0]])})])


def test_a_newcomer_must_be_a_new_agent_object() -> None:
    existing = OneRandomTry()
    with pytest.raises(ValueError, match="born twice"):
        nk.run(LAND, [existing], 2, rng=1, rules=[Scripted({1: ([], [ONES])}, lambda: existing)])
    with pytest.raises(TypeError, match="not an Agent"):
        nk.run(LAND, [OneRandomTry()], 2, rng=1, rules=[Scripted({1: ([], [ONES])}, lambda: "x")])  # type: ignore[arg-type,return-value]


def test_a_rule_must_return_one_row_per_agent_existing_or_not() -> None:
    class ForgetsTheDead(nk.PopulationRule):
        def decide(
            self,
            population: nk.Population,
            proposals: Proposals,
            landscape: nk.Landscape,
            rng: Rng,
        ) -> nk.PopulationDecision:
            living = np.flatnonzero(population.alive)
            dies = [0] if population.period == 1 else []
            return nk.PopulationDecision([proposals[a][0] for a in living], dies=dies)

    with pytest.raises(ValueError, match="must return 3 configurations"):
        nk.run(LAND, [OneRandomTry() for _ in range(3)], 3, rng=1, rules=[ForgetsTheDead()])
