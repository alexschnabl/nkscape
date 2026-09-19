"""
The population: what agents can see of each other, and rules applied to all of them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.landscape import Landscape


@dataclass(frozen=True)
class Population:
    """
    Where every agent was at the end of the last period, and what that was worth.

    Row a belongs to agent a. Both arrays are read-only. Every agent and the interaction rule
    get the same snapshot, so nobody sees a move made in the current period.
    """

    configs: NDArray[np.int8]
    fitness: NDArray[np.floating]


class Interaction(ABC):
    """
    A rule applied to the whole population once per period, after every agent has stepped.

    It covers what is done to agents rather than decided by them: selection, vetoes, a central
    choice among proposals. It may keep memory between periods in `self`.
    """

    @abstractmethod
    def apply(
        self,
        population: Population,
        proposals: tuple[NDArray[np.int8], ...],
        landscape: Landscape,
        rng: np.random.Generator,
    ) -> ArrayLike:
        """
        The configurations the agents really end up with, one row per agent.

        Parameters
        ----------
        population
            Where every agent was before this period.
        proposals
            What every agent's `step` returned. `proposals[a]` holds the proposals of agent a,
            one per row, best first; `proposals[a][0]` is its first choice. Agents may send
            different numbers of proposals. Read-only.
        landscape
            The landscape.
        rng
            The interaction's own random stream.
        """

    def record(self) -> dict[str, float]:
        """
        Extra values to record in every period, by name. The default records nothing.

        The names must be the same in every period.
        """
        return {}
