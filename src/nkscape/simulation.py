"""
Running agents on a landscape.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.agent import Agent
from nkscape.configuration import check_configuration
from nkscape.landscape import Landscape

CUBE_IS_CHEAP_UP_TO = 20


@dataclass(frozen=True)
class RunResult:
    """
    Where an agent was in every period, and what that was worth.

    Row 0 is the start; row t is the state after period t. `extra` holds whatever the agent's
    `record` returned, one array per name.
    """

    configs: NDArray[np.int_]
    fitness: NDArray[np.floating]
    extra: dict[str, NDArray[np.float64]]


def run(
    landscape: Landscape,
    agent: Agent,
    periods: int,
    rng: int | np.random.Generator,
    start: ArrayLike | None = None,
    build_cube: bool | None = None,
) -> RunResult:
    """
    Let one agent walk a landscape for a number of periods.

    Parameters
    ----------
    landscape
        The landscape to walk.
    agent
        The agent; its `step` is called once per period.
    periods
        How many periods to run.
    rng
        A seed, or a generator. It is split into one stream for a random start and one for the
        agent, so giving `start` does not change the agent's randomness.
    start
        Where the agent starts. Drawn at random if not given.
    build_cube
        Whether to build the landscape's fitness cube first, which makes every period about ten
        times faster. By default it is built for N up to 20, where that costs little. `True`
        builds it for any N within the enumeration limit, `False` never builds it.
    """
    if periods < 0:
        raise ValueError(f"periods must not be negative, got {periods}")
    if build_cube or (build_cube is None and landscape.n <= CUBE_IS_CHEAP_UP_TO):
        landscape.fitness_cube()

    start_rng, agent_rng = np.random.default_rng(rng).spawn(2)
    if start is None:
        config = landscape.random_configuration(start_rng)
    else:
        config = check_configuration(start, landscape.n)

    configs = np.empty((periods + 1, landscape.n), dtype=int)
    fitness = np.empty(periods + 1, dtype=landscape.tables.dtype)
    extra = {name: np.empty(periods + 1) for name in agent.record()}

    for t in range(periods + 1):
        if t > 0:
            config.setflags(write=False)
            config = check_configuration(agent.step(config, landscape, agent_rng), landscape.n)
        configs[t] = config
        fitness[t] = landscape.fitness(config)
        recorded = agent.record()
        if recorded.keys() != extra.keys():
            raise ValueError(
                f"record() must return the same names in every period: started with "
                f"{sorted(extra)}, got {sorted(recorded)} in period {t}"
            )
        for name, value in recorded.items():
            extra[name][t] = value

    for array in (configs, fitness, *extra.values()):
        array.setflags(write=False)
    return RunResult(configs, fitness, extra)
