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
Config = NDArray[np.int8]
Proposals = tuple[NDArray[np.int8], ...]
Rng = np.random.Generator


def labels(configs: NDArray[np.int8]) -> list[str]:
    return ["".join(map(str, config)) for config in configs]


class BestNeighbour(nk.Agent):
    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        options = landscape.neighbour_fitness(config)
        j = int(np.argmax(options))
        best: NDArray[np.int_] = landscape.neighbours(config)[j]
        return best if options[j] > landscape.fitness(config) else config


class OneRandomTry(nk.Agent):
    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        candidate: NDArray[np.int_] = landscape.neighbours(config)[rng.integers(landscape.n)]
        return candidate if landscape.fitness(candidate) > landscape.fitness(config) else config


class FlipsDecisionZero(nk.Agent):
    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        flipped: NDArray[np.int_] = landscape.neighbours(config)[0]
        return flipped


class CopiesAgentZero(nk.Agent):
    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        first: NDArray[np.int8] = population.configs[0]
        return first


class CopiesTheBestIfBelowAverage(OneRandomTry):
    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        if landscape.fitness(config) < population.fitness.mean():
            best: NDArray[np.int8] = population.configs[population.fitness.argmax()]
            return best
        return super().step(config, landscape, rng, population)


class CountsItsPeriods(OneRandomTry):
    def __init__(self) -> None:
        self.periods_seen = 0

    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        self.periods_seen += 1
        return super().step(config, landscape, rng, population)

    def record(self) -> dict[str, float]:
        return {"periods_seen": self.periods_seen}


class EditsItsInput(nk.Agent):
    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        config[0] = 1 - config[0]
        return config


class EditsThePopulation(nk.Agent):
    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        population.fitness[0] = 1.0
        return config


class Returns(nk.Agent):
    def __init__(self, answer: ArrayLike) -> None:
        self.answer = answer

    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        return self.answer


class ChangesWhatItRecords(OneRandomTry):
    def __init__(self) -> None:
        self.calls = 0

    def record(self) -> dict[str, float]:
        self.calls += 1
        return {"a": 1.0} if self.calls < 3 else {"b": 1.0}


class ImitateTheBestEveryFifthPeriod(nk.PopulationRule):
    def decide(
        self,
        population: nk.Population,
        proposals: Proposals,
        landscape: nk.Landscape,
        rng: Rng,
    ) -> ArrayLike:
        first_choices = [ranked[0] for ranked in proposals]
        if population.period % 5:
            return first_choices
        best = max(first_choices, key=landscape.fitness)
        return np.tile(best, (len(proposals), 1))


class DrawsRandomNumbersButChangesNothing(nk.PopulationRule):
    def decide(
        self,
        population: nk.Population,
        proposals: Proposals,
        landscape: nk.Landscape,
        rng: Rng,
    ) -> ArrayLike:
        rng.random(7)
        return [ranked[0] for ranked in proposals]


class AppliesWith(nk.PopulationRule):
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def decide(
        self,
        population: nk.Population,
        proposals: Proposals,
        landscape: nk.Landscape,
        rng: Rng,
    ) -> ArrayLike:
        first_choices = np.array([ranked[0] for ranked in proposals])
        if self.answer == "too few rows":
            return first_choices[:-1]
        if self.answer == "a two":
            return first_choices + 2
        proposals[0][0, 0] = 1 - proposals[0][0, 0]
        return first_choices


class SendsItsTwoBestNeighbours(nk.Agent):
    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        order = np.argsort(landscape.neighbour_fitness(config))[::-1]
        return landscape.neighbours(config)[order[:2]]


class ChoosesTheBestProposalOfAnyone(nk.PopulationRule):
    def __init__(self) -> None:
        self.shapes: list[list[tuple[int, ...]]] = []
        self.overruled = 0

    def decide(
        self,
        population: nk.Population,
        proposals: Proposals,
        landscape: nk.Landscape,
        rng: Rng,
    ) -> ArrayLike:
        self.shapes.append([ranked.shape for ranked in proposals])
        everything = np.concatenate(proposals)
        best = max(everything, key=landscape.fitness)
        self.overruled = sum(not np.array_equal(ranked[0], best) for ranked in proposals)
        return np.tile(best, (len(proposals), 1))

    def record(self) -> dict[str, float]:
        return {"overruled": self.overruled}


class RemembersWhatItWasTold(OneRandomTry):
    def __init__(self) -> None:
        self.wanted: list[str] = []
        self.got: list[str] = []
        self.fitness_seen: list[float] = []

    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        wanted = np.asarray(super().step(config, landscape, rng, population))
        self.wanted.append("".join(map(str, wanted)))
        return wanted

    def after_period(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> None:
        self.got.append("".join(map(str, config)))
        self.fitness_seen.append(float(population.fitness.max()))

    def record(self) -> dict[str, float]:
        return {"told": len(self.got)}


class AnnouncesBeforeItSteps(OneRandomTry):
    def __init__(self, board: list[int]) -> None:
        self.board = board
        self.announcements_seen: list[int] = []
        self.start_fitness: float | None = None

    def before_period(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> None:
        self.board.append(1)
        if self.start_fitness is None:
            self.start_fitness = landscape.fitness(config)

    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        self.announcements_seen.append(len(self.board))
        return super().step(config, landscape, rng, population)


class NotesThePeriodEverywhere(OneRandomTry):
    def __init__(self) -> None:
        self.before: list[int] = []
        self.during: list[int] = []
        self.after: list[int] = []

    def before_period(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> None:
        self.before.append(population.period)

    def step(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> ArrayLike:
        self.during.append(population.period)
        return super().step(config, landscape, rng, population)

    def after_period(
        self, config: Config, landscape: nk.Landscape, rng: Rng, population: nk.Population
    ) -> None:
        self.after.append(population.period)


class FlipsDecisionZeroForEveryone(nk.PopulationRule):
    def __init__(self, name: str = "flips") -> None:
        self.name = name
        self.periods: list[int] = []
        self.shapes: list[list[tuple[int, ...]]] = []
        self.draws: list[float] = []

    def decide(
        self,
        population: nk.Population,
        proposals: Proposals,
        landscape: nk.Landscape,
        rng: Rng,
    ) -> ArrayLike:
        self.periods.append(population.period)
        self.shapes.append([ranked.shape for ranked in proposals])
        self.draws.append(float(rng.random()))
        first_choices = np.array([ranked[0] for ranked in proposals])
        first_choices[:, 0] = 1 - first_choices[:, 0]
        return first_choices

    def record(self) -> dict[str, float]:
        return {self.name: len(self.periods)}


class RuleThatChangesWhatItRecords(ChoosesTheBestProposalOfAnyone):
    def record(self) -> dict[str, float]:
        return {"a": 1.0} if len(self.shapes) < 2 else {"b": 1.0}


THREE_STARTS = [[0, 1, 0], [0, 1, 0], [1, 0, 0]]


def three_agents() -> list[nk.Agent]:
    return [BestNeighbour(), OneRandomTry(), OneRandomTry()]


def test_three_agents_on_the_firm_without_a_rule() -> None:
    result = nk.run(FIRM, three_agents(), periods=6, rng=3, starts=THREE_STARTS)
    assert labels(result.configs[0]) == ["010", "000", "000", "000", "000", "000", "000"]
    assert labels(result.configs[1]) == ["010", "011", "001", "101", "101", "111", "111"]
    assert labels(result.configs[2]) == ["100", "110", "110", "110", "110", "110", "110"]
    assert result.fitness[0] == pytest.approx([1.0 / 3] + [2.1 / 3] * 6)
    assert result.extra == {}


def test_a_population_rule_changes_what_is_adopted() -> None:
    rule = ImitateTheBestEveryFifthPeriod()
    result = nk.run(FIRM, three_agents(), periods=6, rng=3, starts=THREE_STARTS, rules=[rule])
    assert labels(result.configs[0]) == ["010", "000", "000", "000", "000", "111", "111"]
    assert labels(result.configs[1]) == ["010", "011", "001", "101", "101", "111", "111"]
    assert labels(result.configs[2]) == ["100", "110", "110", "110", "110", "111", "111"]


def test_without_a_rule_the_first_of_several_proposals_is_adopted() -> None:
    result = nk.run(FIRM, [SendsItsTwoBestNeighbours()], periods=1, rng=1, starts=[[0, 1, 0]])
    assert labels(result.configs[0]) == ["010", "000"]


def test_a_rule_gets_every_proposal_and_may_choose_among_them() -> None:
    rule = ChoosesTheBestProposalOfAnyone()
    agents = [SendsItsTwoBestNeighbours(), BestNeighbour()]
    result = nk.run(FIRM, agents, periods=2, rng=1, starts=[[0, 1, 0], [1, 0, 1]], rules=[rule])
    assert rule.shapes == [[(2, 3), (1, 3)]] * 2
    assert labels(result.configs[0]) == ["010", "111", "111"]
    assert labels(result.configs[1]) == ["101", "111", "111"]
    assert result.rule_extra["overruled"].tolist() == [0, 1, 1]


@pytest.mark.parametrize("answer", [[[1, 0, 1], [1, 0]], [[[1, 0, 1]]], [[1, 0, 1], [1, 2, 1]]])
def test_invalid_proposals_are_rejected(answer: ArrayLike) -> None:
    with pytest.raises(ValueError, match=r"expected|only contain 0 and 1|inhomogeneous"):
        nk.run(FIRM, [Returns(answer)], periods=1, rng=1)


def test_after_period_tells_the_agent_what_it_really_got() -> None:
    agents = [RemembersWhatItWasTold(), RemembersWhatItWasTold(), RemembersWhatItWasTold()]
    rule = ImitateTheBestEveryFifthPeriod()
    result = nk.run(FIRM, agents, periods=6, rng=3, starts=THREE_STARTS, rules=[rule])
    for a, agent in enumerate(agents):
        assert agent.got == labels(result.configs[a])[1:]
        assert agent.fitness_seen == result.fitness.max(axis=0)[1:].tolist()
    assert agents[2].wanted[4] == "110"
    assert agents[2].got[4] == "111"
    assert result.extra["told"].tolist() == [[0, 1, 2, 3, 4, 5, 6]] * 3


def test_every_before_period_runs_before_any_agent_steps() -> None:
    board: list[int] = []
    agents = [AnnouncesBeforeItSteps(board) for _ in range(3)]
    result = nk.run(FIRM, agents, periods=4, rng=1)
    for a, agent in enumerate(agents):
        assert agent.announcements_seen == [3, 6, 9, 12]
        assert agent.start_fitness == result.fitness[a, 0]
    assert len(board) == 3 * 4


def test_every_hook_and_every_rule_is_told_the_period_being_played() -> None:
    agent, rule = NotesThePeriodEverywhere(), FlipsDecisionZeroForEveryone()
    nk.run(FIRM, [agent], periods=4, rng=1, rules=[rule])
    assert agent.before == agent.during == agent.after == rule.periods == [1, 2, 3, 4]


def test_rules_are_applied_one_after_the_other() -> None:
    best, flip = ChoosesTheBestProposalOfAnyone(), FlipsDecisionZeroForEveryone()
    agents = [SendsItsTwoBestNeighbours(), BestNeighbour()]
    starts = [[0, 0, 0], [0, 0, 0]]

    result = nk.run(FIRM, agents, periods=1, rng=1, starts=starts, rules=[best, flip])
    assert labels(result.configs[:, 1]) == ["100", "100"]
    assert best.shapes == [[(2, 3), (1, 3)]]
    assert flip.shapes == [[(1, 3), (1, 3)]]
    assert result.rule_extra["overruled"].tolist() == [0, 1]
    assert result.rule_extra["flips"].tolist() == [0, 1]

    other_order = nk.run(
        FIRM,
        [SendsItsTwoBestNeighbours(), BestNeighbour()],
        periods=1,
        rng=1,
        starts=starts,
        rules=[FlipsDecisionZeroForEveryone(), ChoosesTheBestProposalOfAnyone()],
    )
    assert labels(other_order.configs[:, 1]) == ["101", "101"]


def test_every_rule_has_its_own_random_stream() -> None:
    alone = FlipsDecisionZeroForEveryone()
    nk.run(FIRM, three_agents(), periods=3, rng=9, rules=[alone])

    first, second = FlipsDecisionZeroForEveryone("a"), FlipsDecisionZeroForEveryone("b")
    nk.run(FIRM, three_agents(), periods=3, rng=9, rules=[first, second])
    assert first.draws == alone.draws
    assert second.draws != first.draws


def test_a_bare_rule_instead_of_a_list_is_rejected_clearly() -> None:
    with pytest.raises(TypeError, match=r"pass \[rule\]"):
        nk.run(FIRM, three_agents(), periods=1, rng=1, rules=FlipsDecisionZeroForEveryone())  # type: ignore[arg-type]


def test_the_same_rule_object_twice_is_rejected() -> None:
    rule = FlipsDecisionZeroForEveryone()
    with pytest.raises(ValueError, match="every rule must be its own object"):
        nk.run(FIRM, three_agents(), periods=1, rng=1, rules=[rule, rule])


def test_two_rules_recording_the_same_name_are_rejected() -> None:
    rules = [FlipsDecisionZeroForEveryone(), FlipsDecisionZeroForEveryone()]
    with pytest.raises(ValueError, match="two rules record the same name 'flips'"):
        nk.run(FIRM, three_agents(), periods=1, rng=1, rules=rules)


def test_a_rule_without_record_leaves_rule_extra_empty() -> None:
    rule = ImitateTheBestEveryFifthPeriod()
    result = nk.run(FIRM, three_agents(), periods=2, rng=3, rules=[rule])
    assert result.rule_extra == {}


def test_a_rule_that_changes_its_recorded_names_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"rule 0 started with \['a'\].*\['b'\]"):
        nk.run(FIRM, three_agents(), periods=5, rng=1, rules=[RuleThatChangesWhatItRecords()])


def test_result_shapes_put_the_agent_first() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    result = nk.run(land, [OneRandomTry() for _ in range(7)], periods=30, rng=7)
    assert result.configs.shape == (7, 31, 10)
    assert result.fitness.shape == (7, 31)
    assert result.configs.dtype == np.int8
    assert result.fitness.mean(axis=0).shape == (31,)


def test_zero_periods_records_only_the_start() -> None:
    result = nk.run(FIRM, [BestNeighbour()], periods=0, rng=1, starts=[[0, 1, 0]])
    assert labels(result.configs[0]) == ["010"]
    assert result.fitness.shape == (1, 1)


def test_an_agent_walks_the_same_way_alone_as_in_a_crowd() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    alone = nk.run(land, [OneRandomTry()], periods=40, rng=5)
    crowd = nk.run(land, [OneRandomTry() for _ in range(50)], periods=40, rng=5)
    assert np.array_equal(alone.configs[0], crowd.configs[0])
    assert not np.array_equal(crowd.configs[0], crowd.configs[1])


def test_everybody_decides_on_last_periods_state() -> None:
    result = nk.run(
        FIRM, [FlipsDecisionZero(), CopiesAgentZero()], periods=4, rng=1, starts=[[0, 0, 0]] * 2
    )
    assert labels(result.configs[0]) == ["000", "100", "000", "100", "000"]
    assert labels(result.configs[1]) == ["000", "000", "100", "000", "100"]


def test_an_agent_can_decide_from_what_it_sees_of_the_others() -> None:
    land = nk.Landscape.random_k(10, 3, rng=2)
    result = nk.run(land, [CopiesTheBestIfBelowAverage() for _ in range(30)], periods=20, rng=3)
    assert len(set(labels(result.configs[:, 0]))) > 20
    assert len(set(labels(result.configs[:, -1]))) == 1


def test_same_seed_gives_the_same_run() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    a = nk.run(land, [OneRandomTry() for _ in range(5)], periods=30, rng=7)
    b = nk.run(land, [OneRandomTry() for _ in range(5)], periods=30, rng=7)
    c = nk.run(land, [OneRandomTry() for _ in range(5)], periods=30, rng=8)
    assert np.array_equal(a.configs, b.configs)
    assert np.array_equal(a.fitness, b.fitness)
    assert not np.array_equal(a.configs, c.configs)


def test_giving_the_starts_does_not_change_the_agents_randomness() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    drawn = nk.run(land, [OneRandomTry() for _ in range(5)], periods=30, rng=7)
    given = nk.run(
        land, [OneRandomTry() for _ in range(5)], periods=30, rng=7, starts=drawn.configs[:, 0]
    )
    assert np.array_equal(drawn.configs, given.configs)


def test_the_rules_randomness_does_not_disturb_the_agents() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    rule = DrawsRandomNumbersButChangesNothing()
    without = nk.run(land, [OneRandomTry() for _ in range(5)], periods=30, rng=7)
    with_rule = nk.run(land, [OneRandomTry() for _ in range(5)], 30, rng=7, rules=[rule])
    assert np.array_equal(without.configs, with_rule.configs)


def test_a_generator_can_be_passed() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    streams = np.random.default_rng(42).spawn(2)
    a = nk.run(land, [OneRandomTry()], periods=30, rng=streams[0])
    b = nk.run(land, [OneRandomTry()], periods=30, rng=streams[1])
    assert not np.array_equal(a.configs, b.configs)

    replay = np.random.default_rng(42).spawn(2)
    assert np.array_equal(nk.run(land, [OneRandomTry()], 30, replay[0]).configs, a.configs)


def test_recorded_fitness_is_the_fitness_of_the_recorded_configuration() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1)
    result = nk.run(land, [OneRandomTry() for _ in range(3)], periods=20, rng=7)
    for a in range(3):
        for config, fitness in zip(result.configs[a], result.fitness[a], strict=True):
            assert land.fitness(config) == fitness


def test_agents_keep_their_memory_and_can_record_it() -> None:
    agents = [CountsItsPeriods(), CountsItsPeriods()]
    result = nk.run(FIRM, agents, periods=5, rng=1)
    assert agents[0].periods_seen == 5
    assert result.extra["periods_seen"].tolist() == [[0, 1, 2, 3, 4, 5]] * 2


def test_an_agent_that_does_not_record_a_name_gets_nan() -> None:
    result = nk.run(FIRM, [CountsItsPeriods(), OneRandomTry()], periods=3, rng=1)
    assert result.extra["periods_seen"][0].tolist() == [0, 1, 2, 3]
    assert np.isnan(result.extra["periods_seen"][1]).all()


def test_changing_the_recorded_names_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"same names.*agent 0.*\['a'\].*\['b'\]"):
        nk.run(FIRM, [ChangesWhatItRecords()], periods=5, rng=1)


def test_the_same_agent_object_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="its own object"):
        nk.run(FIRM, [OneRandomTry()] * 3, periods=2, rng=1)


def test_a_bare_agent_instead_of_a_list_is_rejected_clearly() -> None:
    with pytest.raises(TypeError, match=r"pass \[agent\]"):
        nk.run(FIRM, OneRandomTry(), periods=2, rng=1)  # type: ignore[arg-type]


def test_no_agents_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one agent"):
        nk.run(FIRM, [], periods=2, rng=1)


def test_negative_periods_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        nk.run(FIRM, [BestNeighbour()], periods=-1, rng=1)


def test_starts_must_have_one_row_per_agent() -> None:
    with pytest.raises(ValueError, match="expected 3 configurations of 3 values"):
        nk.run(FIRM, three_agents(), periods=2, rng=1, starts=[[0, 1, 0], [0, 1, 0]])


@pytest.mark.parametrize("agent", [EditsItsInput(), EditsThePopulation()])
def test_an_agent_that_edits_what_it_is_shown_is_stopped(agent: nk.Agent) -> None:
    with pytest.raises(ValueError, match="read-only"):
        nk.run(FIRM, [agent], periods=2, rng=1)


@pytest.mark.parametrize("answer", [[1, 0], [1, 2, 0]])
def test_an_agent_that_returns_no_valid_configuration_is_stopped(answer: ArrayLike) -> None:
    with pytest.raises(ValueError, match=r"expected 3 values|only contain 0 and 1"):
        nk.run(FIRM, [Returns(answer)], periods=2, rng=1)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ("too few rows", "expected 3 configurations"),
        ("a two", "0 and 1"),
        ("edits its input", "read-only"),
    ],
)
def test_a_rule_that_misbehaves_is_stopped(answer: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        nk.run(FIRM, three_agents(), periods=2, rng=1, rules=[AppliesWith(answer)])


def test_agent_and_rule_cannot_be_created_without_their_method() -> None:
    class ForgotStep(nk.Agent):
        pass

    class ForgotDecide(nk.PopulationRule):
        pass

    with pytest.raises(TypeError, match="step"):
        ForgotStep()  # type: ignore[abstract]
    with pytest.raises(TypeError, match="decide"):
        ForgotDecide()  # type: ignore[abstract]


def test_results_are_read_only() -> None:
    result = nk.run(FIRM, [CountsItsPeriods()], periods=2, rng=1)
    for array in (result.configs, result.fitness, result.extra["periods_seen"]):
        assert not array.flags.writeable


def test_a_small_landscape_gets_its_cube_even_for_a_tiny_run() -> None:
    land = nk.Landscape.next_k(8, 2, rng=1)
    nk.run(land, [OneRandomTry()], periods=1, rng=1)
    assert land._fitness_cube is not None


def test_the_cube_is_built_only_when_the_run_is_long_enough_to_repay_it() -> None:
    needed = 2**20 * 20 / 40_000
    assert 100 < needed < 600

    short = nk.Landscape.next_k(20, 2, rng=1)
    nk.run(short, [OneRandomTry()], periods=100, rng=1)
    assert short._fitness_cube is None

    long = nk.Landscape.next_k(20, 2, rng=1)
    nk.run(long, [OneRandomTry() for _ in range(6)], periods=100, rng=1)
    assert long._fitness_cube is not None


def test_the_cube_is_never_built_automatically_above_the_enumeration_limit() -> None:
    land = nk.Landscape.next_k(6, 2, rng=1)
    nk.set_enumeration_limit(5)
    try:
        nk.run(land, [OneRandomTry() for _ in range(3)], periods=50, rng=1)
    finally:
        nk.set_enumeration_limit(24)
    assert land._fitness_cube is None


def test_build_cube_can_be_forced_or_forbidden() -> None:
    small = nk.Landscape.next_k(8, 2, rng=1)
    large = nk.Landscape.next_k(21, 2, rng=1)
    nk.run(small, [OneRandomTry()], periods=1, rng=1, build_cube=False)
    nk.run(large, [OneRandomTry()], periods=1, rng=1, build_cube=True)
    assert small._fitness_cube is None
    assert large._fitness_cube is not None


def test_forcing_the_cube_above_the_enumeration_limit_is_refused() -> None:
    land = nk.Landscape.next_k(25, 2, rng=1)
    with pytest.raises(ValueError, match="above the limit of 24"):
        nk.run(land, [OneRandomTry()], periods=1, rng=1, build_cube=True)


def test_a_run_gives_the_same_walk_with_and_without_the_cube() -> None:
    agents = [OneRandomTry() for _ in range(4)]
    with_cube = nk.run(nk.Landscape.random_k(10, 3, rng=1), agents, 40, rng=5)
    without = nk.run(nk.Landscape.random_k(10, 3, rng=1), agents, 40, rng=5, build_cube=False)
    assert np.array_equal(with_cube.configs, without.configs)
    assert np.array_equal(with_cube.fitness, without.fitness)


def test_a_float32_landscape_records_float32_fitness() -> None:
    land = nk.Landscape.random_k(10, 3, rng=1, dtype=np.float32)
    assert nk.run(land, [OneRandomTry()], periods=5, rng=1).fitness.dtype == np.float32
