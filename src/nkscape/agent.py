"""
The agent: a rule for walking a landscape.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.landscape import Landscape


class Agent(ABC):
    """
    Something that walks a landscape: given where it is, it says where it wants to be next.

    Subclass it and write `step`. An agent may keep memory between periods in `self`.
    """

    @abstractmethod
    def step(
        self, config: NDArray[np.int_], landscape: Landscape, rng: np.random.Generator
    ) -> ArrayLike:
        """
        The configuration the agent wants to be at in the next period.

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
        """

    def record(self) -> dict[str, float]:
        """
        Extra values to record in every period, by name. The default records nothing.

        The names must be the same in every period.
        """
        return {}
