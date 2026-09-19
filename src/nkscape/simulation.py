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
from nkscape.population import Interaction, Population

CUBE_CELLS_WORTH_ONE_AGENT_PERIOD = 40_000


@dataclass(frozen=True)
class RunResult:
    """
    Where every agent was in every period, and what that was worth.

    `configs[a, t]` is the configuration of agent a after period t and `fitness[a, t]` its
    fitness; t = 0 is the start. `extra[name][a, t]` holds what the agents' `record` returned,
    NaN where an agent does not record that name. `interaction_extra[name][t]` holds what the
    interaction rule's `record` returned. All arrays are read-only.
    """

    configs: NDArray[np.int8]
    fitness: NDArray[np.floating]
    extra: dict[str, NDArray[np.float64]]
    interaction_extra: dict[str, NDArray[np.float64]]


def run(
    landscape: Landscape,
    agents: Sequence[Agent],
    periods: int,
    rng: int | np.random.Generator,
    starts: ArrayLike | None = None,
    interaction: Interaction | None = None,
    build_cube: bool | None = None,
) -> RunResult:
    """
    Let a population of agents walk a landscape for a number of periods.

    In every period all agents step, each seeing where everybody was at the end of the last
    period; then the interaction rule, if any, decides what is really adopted; then every agent's
    `after_period` is told the outcome.

    Parameters
    ----------
    landscape
        The landscape to walk.
    agents
        The agents, each its own object. A single agent is a list of one.
    periods
        How many periods to run.
    rng
        A seed, or a generator. It is split into one stream for the interaction rule and, for
        every agent, one stream for its random start and one for its steps. Without an
        interaction rule an agent therefore walks the same way alone as in a crowd.
    starts
        Where the agents start, one row per agent. Drawn at random if not given.
    interaction
        A rule applied to the whole population after every agent has stepped. Without one, every
        agent gets its first choice.
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

    n, count = landscape.n, len(agents)
    work = count * periods * CUBE_CELLS_WORTH_ONE_AGENT_PERIOD
    pays_off = n <= get_enumeration_limit() and work >= 2**n * n
    if build_cube or (build_cube is None and pays_off):
        landscape.fitness_cube()

    interaction_rng, agents_rng = np.random.default_rng(rng).spawn(2)
    streams = [stream.spawn(2) for stream in agents_rng.spawn(count)]
    recorders: list[Agent | Interaction] = [*agents]
    if interaction is not None:
        recorders.append(interaction)

    configs = np.empty((count, periods + 1, n), dtype=np.int8)
    fitness = np.empty((count, periods + 1), dtype=landscape.tables.dtype)
    extra: dict[str, NDArray[np.float64]] = {}
    interaction_extra: dict[str, NDArray[np.float64]] = {}
    names: list[set[str]] = []

    if starts is None:
        for a in range(count):
            configs[a, 0] = landscape.random_configuration(streams[a][0])
    else:
        configs[:, 0] = check_configuration(starts, n, count)

    population = Population(configs[:, 0], fitness[:, 0])
    for t in range(periods + 1):
        if t > 0:
            proposals: list[NDArray[np.int8]] = []
            for a, agent in enumerate(agents):
                here, step_rng = population.configs[a], streams[a][1]
                wanted = np.asarray(agent.step(here, landscape, step_rng, population))
                ranked = check_configuration(wanted, n, len(wanted) if wanted.ndim == 2 else None)
                proposal = ranked.reshape(-1, n).astype(np.int8)
                proposal.setflags(write=False)
                proposals.append(proposal)

            if interaction is None:
                configs[:, t] = [proposal[0] for proposal in proposals]
            else:
                adopted = interaction.apply(
                    population, tuple(proposals), landscape, interaction_rng
                )
                configs[:, t] = check_configuration(adopted, n, count)

        for a in range(count):
            fitness[a, t] = landscape.fitness(configs[a, t])
        now_configs, now_fitness = configs[:, t], fitness[:, t]
        now_configs.setflags(write=False)
        now_fitness.setflags(write=False)
        population = Population(now_configs, now_fitness)

        if t > 0:
            for a, agent in enumerate(agents):
                agent.after_period(now_configs[a], landscape, streams[a][1], population)

        for r, recorder in enumerate(recorders):
            recorded = recorder.record()
            who = f"agent {r}" if r < count else "the interaction rule"
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
                    if name not in interaction_extra:
                        interaction_extra[name] = np.full(periods + 1, np.nan)
                    interaction_extra[name][t] = value

    for array in (configs, fitness, *extra.values(), *interaction_extra.values()):
        array.setflags(write=False)
    return RunResult(configs, fitness, extra, interaction_extra)
