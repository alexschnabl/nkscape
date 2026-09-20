"""
The hill climber: local search over single-decision changes, optionally with long jumps.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.agent import Agent
from nkscape.landscape import Landscape
from nkscape.population import Population


class HillClimber(Agent):
    """
    Moves, once per period, to a neighbour that is strictly better, and stays if there is none.

    With `pick="first"` and `long_jumps=True` this is the search of Levinthal (1997).

    Parameters
    ----------
    pick
        Which better neighbour is taken.

        - `"first"`: the first one found when the decisions are tried in order 0, 1, 2, ...
          Uses no randomness. Low-numbered decisions are therefore changed first, which makes
          agents alike sooner than `"random"` does.
        - `"random"`: one of the better neighbours, each equally likely. The same as taking the
          first one found when the decisions are tried in random order.
        - `"best"`: the one with the highest fitness (steepest ascent). Exact ties are broken
          at random.
    long_jumps
        Whether the agent also draws one configuration uniformly at random every period.

        - `False` (default): local search only.
        - `True`: the draw is taken instead if it is strictly better than the neighbour picked,
          or than where the agent stands when no neighbour is better.

    References
    ----------
    Levinthal, D. A. (1997). Adaptation on rugged landscapes. Management Science, 43(7),
    934-950. https://doi.org/10.1287/mnsc.43.7.934
    """

    def __init__(self, pick: Literal["first", "random", "best"], long_jumps: bool = False) -> None:
        if pick not in ("first", "random", "best"):
            raise ValueError(f'pick must be "first", "random" or "best", got {pick!r}')
        self.pick = pick
        self.long_jumps = long_jumps

    def step(
        self,
        config: NDArray[np.int8],
        landscape: Landscape,
        rng: np.random.Generator,
        population: Population,
    ) -> ArrayLike:
        """
        The configuration the agent wants next: a better neighbour, a better long jump, or `config`.
        """
        current = landscape.fitness(config)
        around = landscape.neighbour_fitness(config)
        better = np.flatnonzero(around > current)
        next_config, next_fitness = config, current
        if better.size > 0:
            if self.pick == "first":
                flip = better[0]
            elif self.pick == "random":
                flip = rng.choice(better)
            else:
                flip = rng.choice(np.flatnonzero(around == around.max()))
            next_config = config.copy()
            next_config[flip] ^= 1
            next_fitness = around[flip]
        if self.long_jumps:
            jump = landscape.random_configuration(rng)
            if landscape.fitness(jump) > next_fitness:
                return jump
        return next_config
