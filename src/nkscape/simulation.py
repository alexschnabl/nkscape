"""
Running agents on a landscape.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.agent import Agent
from nkscape.configuration import check_configuration
from nkscape.landscape import Landscape, get_enumeration_limit
from nkscape.population import Population, PopulationRule

CUBE_CELLS_WORTH_ONE_AGENT_PERIOD = 40_000


@dataclass(frozen=True)
class RunResult:
    """
    Where every agent was in every period, and what that was worth.

    `configs[a, t]` is the configuration of agent a after period t and `fitness[a, t]` its
    fitness; t = 0 is the start. `extra[name][a, t]` holds what the agents' `record` returned,
    NaN where an agent does not record that name. `rule_extra[name][t]` holds what the
    population rules' `record` returned. All arrays are read-only.
    """

    configs: NDArray[np.int8]
    fitness: NDArray[np.floating]
    extra: dict[str, NDArray[np.float64]]
    rule_extra: dict[str, NDArray[np.float64]]


def run(
    landscape: Landscape,
    agents: Sequence[Agent],
    periods: int,
    rng: int | np.random.Generator,
    starts: ArrayLike | None = None,
    rules: Sequence[PopulationRule] = (),
    build_cube: bool | None = None,
) -> RunResult:
    """
    Let a population of agents walk a landscape for a number of periods.

    Every period runs in this order: every agent's `before_period`; every agent's `step`, each
    seeing where everybody was at the end of the last period; the population rules, if any,
    decide what is really adopted; every agent's `after_period` is told the outcome.

    Parameters
    ----------
    landscape
        The landscape to walk.
    agents
        The agents, each its own object. A single agent is a list of one.
    periods
        How many periods to run.
    rng
        A seed, or a generator. It is split into one stream for every population rule and, for
        every agent, one stream for its random start and one for its steps. Without population
        rules an agent therefore walks the same way alone as in a crowd.
    starts
        Where the agents start, one row per agent. Drawn at random if not given.
    rules
        Rules applied to the whole population after every agent has stepped, one after the
        other: the first gets the agents' proposals, each later one gets what the rule before it
        decided. Without rules, every agent gets its first choice.
    build_cube
        Whether to build the landscape's fitness cube first, which makes every period about ten
        times faster. By default it is built when the run is long enough to repay the cost of
        building it: agents x periods >= 2^N x N / 40,000 (and N is within the enumeration
        limit). `True` always builds it, `False` never does.
    """
    if periods < 0:
        raise ValueError(f"periods must not be negative, got {periods}")
    if isinstance(agents, Agent):
        raise TypeError("agents must be a list of agents; for a single agent pass [agent]")
    if len(agents) == 0:
        raise ValueError("at least one agent is needed")
    if len({id(agent) for agent in agents}) != len(agents):
        raise ValueError("every agent must be its own object; the same agent was given twice")
    if isinstance(rules, PopulationRule):
        raise TypeError("rules must be a list of rules; for a single rule pass [rule]")
    if len({id(rule) for rule in rules}) != len(rules):
        raise ValueError("every rule must be its own object; the same rule was given twice")

    n, count = landscape.n, len(agents)
    work = count * periods * CUBE_CELLS_WORTH_ONE_AGENT_PERIOD
    pays_off = n <= get_enumeration_limit() and work >= 2**n * n
    if build_cube or (build_cube is None and pays_off):
        landscape.fitness_cube()

    rules_rng, agents_rng = np.random.default_rng(rng).spawn(2)
    rule_streams = rules_rng.spawn(len(rules))
    streams = [stream.spawn(2) for stream in agents_rng.spawn(count)]
    recorders: list[Agent | PopulationRule] = [*agents, *rules]

    configs = np.empty((count, periods + 1, n), dtype=np.int8)
    fitness = np.empty((count, periods + 1), dtype=landscape.tables.dtype)
    extra: dict[str, NDArray[np.float64]] = {}
    rule_extra: dict[str, NDArray[np.float64]] = {}
    names: list[set[str]] = []

    if starts is None:
        for a in range(count):
            configs[a, 0] = landscape.random_configuration(streams[a][0])
    else:
        configs[:, 0] = check_configuration(starts, n, count)

    now_configs, now_fitness = configs[:, 0], fitness[:, 0]
    for t in range(periods + 1):
        if t > 0:
            population = Population(now_configs, now_fitness, t)
            for a, agent in enumerate(agents):
                agent.before_period(population.configs[a], landscape, streams[a][1], population)

            proposals: list[NDArray[np.int8]] = []
            for a, agent in enumerate(agents):
                here, step_rng = population.configs[a], streams[a][1]
                wanted = np.asarray(agent.step(here, landscape, step_rng, population))
                ranked = check_configuration(wanted, n, len(wanted) if wanted.ndim == 2 else None)
                proposal = ranked.reshape(-1, n).astype(np.int8)
                proposal.setflags(write=False)
                proposals.append(proposal)

            configs[:, t] = [proposal[0] for proposal in proposals]
            for rule, rule_rng in zip(rules, rule_streams, strict=True):
                decided = rule.decide(population, tuple(proposals), landscape, rule_rng)
                adopted = check_configuration(decided, n, count).astype(np.int8)
                adopted.setflags(write=False)
                configs[:, t] = adopted
                proposals = [adopted[a : a + 1] for a in range(count)]

        for a in range(count):
            fitness[a, t] = landscape.fitness(configs[a, t])
        now_configs, now_fitness = configs[:, t], fitness[:, t]
        now_configs.setflags(write=False)
        now_fitness.setflags(write=False)

        if t > 0:
            outcome = Population(now_configs, now_fitness, t)
            for a, agent in enumerate(agents):
                agent.after_period(now_configs[a], landscape, streams[a][1], outcome)

        for r, recorder in enumerate(recorders):
            recorded = recorder.record()
            who = f"agent {r}" if r < count else f"rule {r - count}"
            if t == 0:
                names.append(set(recorded))
            elif set(recorded) != names[r]:
                raise ValueError(
                    f"record() must return the same names in every period: {who} started "
                    f"with {sorted(names[r])}, got {sorted(recorded)} in period {t}"
                )
            for name, value in recorded.items():
                if r < count:
                    if name not in extra:
                        extra[name] = np.full((count, periods + 1), np.nan)
                    extra[name][r, t] = value
                else:
                    if t == 0:
                        if name in rule_extra:
                            raise ValueError(f"two rules record the same name {name!r}")
                        rule_extra[name] = np.full(periods + 1, np.nan)
                    rule_extra[name][t] = value

    for array in (configs, fitness, *extra.values(), *rule_extra.values()):
        array.setflags(write=False)
    return RunResult(configs, fitness, extra, rule_extra)
