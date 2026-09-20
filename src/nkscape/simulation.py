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
from nkscape.population import Population, PopulationDecision, PopulationRule

CUBE_CELLS_WORTH_ONE_AGENT_PERIOD = 40_000


@dataclass(frozen=True)
class RunResult:
    """
    Where every agent was in every period, and what that was worth.

    Row a belongs to `agents[a]`; every agent that ever existed keeps its row, including those
    born during the run. Column t is the state after period t; t = 0 is the start. All arrays
    are read-only.

    Attributes
    ----------
    configs
        `configs[a, t]` is the configuration of agent a. All -1 where the agent did not exist.
    fitness
        `fitness[a, t]` is the fitness of agent a. NaN where the agent did not exist.
    alive
        `alive[a, t]` says whether agent a existed.
    extra
        `extra[name][a, t]` holds what the agents' `record` returned. NaN where an agent does
        not record that name or did not exist.
    rule_extra
        `rule_extra[name][t]` holds what the population rules' `record` returned.
    agents
        The agent objects, those born during the run included.
    """

    configs: NDArray[np.int8]
    fitness: NDArray[np.floating]
    alive: NDArray[np.bool_]
    extra: dict[str, NDArray[np.float64]]
    rule_extra: dict[str, NDArray[np.float64]]
    agents: list[Agent]


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

    Every period runs in this order:

    1. Every agent's `before_period`.
    2. Every agent's `step`, each seeing where everybody was at the end of the last period.
    3. The population rules, if any, decide what is really adopted, who dies and who is born.
    4. Every agent's `after_period` is told the outcome.
    5. Every agent's and every rule's `record`.

    Agents that do not exist are never called.

    Parameters
    ----------
    landscape
        The landscape to walk.
    agents
        The agents that exist at the start, each its own object. A single agent is a list of one.
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
        times faster.

        - `None` (default): built when the run is long enough to repay the cost of building it,
          that is when agents x periods >= 2^N x N / 40,000 and N is within the enumeration
          limit.
        - `True`: always built.
        - `False`: never built.
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

    n = landscape.n
    members = list(agents)
    work = len(members) * periods * CUBE_CELLS_WORTH_ONE_AGENT_PERIOD
    pays_off = n <= get_enumeration_limit() and work >= 2**n * n
    if build_cube or (build_cube is None and pays_off):
        landscape.fitness_cube()

    rules_rng, agents_rng = np.random.default_rng(rng).spawn(2)
    rule_streams = rules_rng.spawn(len(rules))
    streams = [stream.spawn(2) for stream in agents_rng.spawn(len(members))]

    configs = np.full((len(members), periods + 1, n), -1, dtype=np.int8)
    fitness = np.full((len(members), periods + 1), np.nan, dtype=landscape.tables.dtype)
    alive = np.zeros((len(members), periods + 1), dtype=bool)
    extra: dict[str, NDArray[np.float64]] = {}
    rule_extra: dict[str, NDArray[np.float64]] = {}
    agent_names: list[set[str]] = []
    rule_names: list[set[str]] = []
    nobody = np.full((1, n), -1, dtype=np.int8)
    nobody.setflags(write=False)

    if starts is None:
        for a in range(len(members)):
            configs[a, 0] = landscape.random_configuration(streams[a][0])
    else:
        configs[:, 0] = check_configuration(starts, n, len(members))
    alive[:, 0] = True

    for t in range(periods + 1):
        survivors = list(range(len(members)))
        if t > 0:
            count = len(members)
            was = Population(
                configs[:count, t - 1], fitness[:count, t - 1], t, alive[:count, t - 1]
            )
            for view in (was.configs, was.fitness, was.alive):
                view.setflags(write=False)
            living = [int(a) for a in np.flatnonzero(was.alive)]

            for a in living:
                members[a].before_period(was.configs[a], landscape, streams[a][1], was)

            proposals: list[NDArray[np.int8]] = [nobody] * count
            for a in living:
                wanted = np.asarray(members[a].step(was.configs[a], landscape, streams[a][1], was))
                ranked = check_configuration(wanted, n, len(wanted) if wanted.ndim == 2 else None)
                proposal = ranked.reshape(-1, n).astype(np.int8)
                proposal.setflags(write=False)
                proposals[a] = proposal

            adopted = np.full((count, n), -1, dtype=np.int8)
            for a in living:
                adopted[a] = proposals[a][0]
            dies: set[int] = set()
            born: list[tuple[Agent, ArrayLike]] = []
            for rule, rule_rng in zip(rules, rule_streams, strict=True):
                decision = rule.decide(was, tuple(proposals), landscape, rule_rng)
                if not isinstance(decision, PopulationDecision):
                    decision = PopulationDecision(decision)
                decided = np.asarray(decision.configs)
                if decided.shape != (count, n):
                    raise ValueError(
                        f"a rule must return {count} configurations of {n} values (one row per "
                        f"agent, existing or not), got shape {decided.shape}"
                    )
                adopted = adopted.copy()
                adopted[living] = check_configuration(decided[living], n, len(living))
                adopted.setflags(write=False)
                for a in decision.dies:
                    if not 0 <= a < count or not was.alive[a]:
                        raise ValueError(f"a rule let agent {a} die, which does not exist")
                    dies.add(int(a))
                born.extend(decision.born)
                proposals = [adopted[a : a + 1] if was.alive[a] else nobody for a in range(count)]

            survivors = [a for a in living if a not in dies]
            for a in survivors:
                configs[a, t] = adopted[a]
                alive[a, t] = True

            for newcomer, start in born:
                if not isinstance(newcomer, Agent):
                    raise TypeError(f"a rule gave birth to {newcomer!r}, which is not an Agent")
                if any(newcomer is agent for agent in members):
                    raise ValueError("every agent must be its own object; one was born twice")
                row = len(members)
                if row == len(configs):
                    configs = np.concatenate([configs, np.full_like(configs, -1)])
                    fitness = np.concatenate([fitness, np.full_like(fitness, np.nan)])
                    alive = np.concatenate([alive, np.zeros_like(alive)])
                    for name, values in extra.items():
                        extra[name] = np.concatenate([values, np.full_like(values, np.nan)])
                members.append(newcomer)
                streams.append(agents_rng.spawn(1)[0].spawn(2))
                configs[row, t] = check_configuration(start, n)
                alive[row, t] = True

        count = len(members)
        for a in np.flatnonzero(alive[:count, t]).tolist():
            fitness[a, t] = landscape.fitness(configs[a, t])

        if t > 0:
            now = Population(configs[:count, t], fitness[:count, t], t, alive[:count, t])
            for view in (now.configs, now.fitness, now.alive):
                view.setflags(write=False)
            for a in survivors:
                members[a].after_period(now.configs[a], landscape, streams[a][1], now)

        for a in np.flatnonzero(alive[:count, t]).tolist():
            recorded = members[a].record()
            if a == len(agent_names):
                agent_names.append(set(recorded))
            elif set(recorded) != agent_names[a]:
                raise ValueError(
                    f"record() must return the same names in every period: agent {a} started "
                    f"with {sorted(agent_names[a])}, got {sorted(recorded)} in period {t}"
                )
            for name, value in recorded.items():
                if name not in extra:
                    extra[name] = np.full((len(configs), periods + 1), np.nan)
                extra[name][a, t] = value

        for r, rule in enumerate(rules):
            recorded = rule.record()
            if t == 0:
                rule_names.append(set(recorded))
                for name in recorded:
                    if name in rule_extra:
                        raise ValueError(f"two rules record the same name {name!r}")
                    rule_extra[name] = np.full(periods + 1, np.nan)
            elif set(recorded) != rule_names[r]:
                raise ValueError(
                    f"record() must return the same names in every period: rule {r} started "
                    f"with {sorted(rule_names[r])}, got {sorted(recorded)} in period {t}"
                )
            for name, value in recorded.items():
                rule_extra[name][t] = value

    total = len(members)
    configs, fitness, alive = configs[:total], fitness[:total], alive[:total]
    extra = {name: values[:total] for name, values in extra.items()}
    for array in (configs, fitness, alive, *extra.values(), *rule_extra.values()):
        array.setflags(write=False)
    return RunResult(configs, fitness, alive, extra, rule_extra, members)
