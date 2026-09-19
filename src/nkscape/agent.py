"""
The agent: a rule for walking a landscape.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.landscape import Landscape
from nkscape.population import Population


class Agent(ABC):
    """
    Something that walks a landscape: given where it is, it says where it wants to be next.

    Subclass it and write `step`. An agent may keep memory between periods in `self`.
    """

    @abstractmethod
    def step(
        self,
        config: NDArray[np.int8],
        landscape: Landscape,
        rng: np.random.Generator,
        population: Population,
    ) -> ArrayLike:
        """
        The configuration the agent wants to be at in the next period.

        Return one configuration, or several in order of preference, best first. Without an
        interaction rule the first is adopted; an interaction rule gets all of them.

        Parameters
        ----------
        config
            Where the agent is now. Read-only: return a new configuration, do not edit this one.
            Returning `config` itself means staying.
        landscape
            What the agent looks around with: `fitness`, `neighbours`, `neighbour_fitness`,
            `is_peak`.
        rng
            The agent's own random stream.
        population
            Where all agents were at the end of the last period, and what that was worth.
        """

    def after_period(
        self,
        config: NDArray[np.int8],
        landscape: Landscape,
        rng: np.random.Generator,
        population: Population,
    ) -> None:
        """
        Called once every agent has stepped and the interaction rule has decided.

        `config` is what the agent really ended up with, which need not be what it wanted, and
        `population` is where everybody is now. The default does nothing.
        """
        return

    def record(self) -> dict[str, float]:
        """
        Extra values to record in every period, by name. The default records nothing.

        The names must be the same in every period.
        """
        return {}
