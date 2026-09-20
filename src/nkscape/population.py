"""
The population: what agents can see of each other, and rules applied to all of them.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nkscape.landscape import Landscape

if TYPE_CHECKING:
    from nkscape.agent import Agent


@dataclass(frozen=True)
class Population:
    """
    The state of the world as the agents and the population rules see it.

    Row a belongs to agent a for the whole run. All arrays are read-only.

    - In `before_period`, `step` and `decide` the arrays show the end of the last period, so
      nobody sees a move made in the current one.
    - In `after_period` they show the outcome of the current period.

    Attributes
    ----------
    configs
        Where every agent stands, one row per agent. All -1 for an agent that does not exist.
    fitness
        What that is worth. NaN for an agent that does not exist.
    period
        The period being played, 1 for the first.
    alive
        Which agents exist.
    """

    configs: NDArray[np.int8]
    fitness: NDArray[np.floating]
    period: int
    alive: NDArray[np.bool_]


@dataclass(frozen=True)
class PopulationDecision:
    """
    What a population rule decides when it also changes who exists.

    Parameters
    ----------
    configs
        The configurations the agents end up with, one row per agent. Rows of agents that do
        not exist are ignored.
    dies
        The agents that stop existing at the end of this period, by row number.
    born
        The agents that start existing at the end of this period: for each, a new agent object
        and its starting configuration.
    """

    configs: ArrayLike
    dies: Sequence[int] = ()
    born: Sequence[tuple["Agent", ArrayLike]] = ()


class PopulationRule(ABC):
    """
    A rule applied to the whole population once per period, after every agent has stepped.

    It covers what is done to agents rather than decided by them: selection, vetoes, a central
    choice among proposals, births and deaths. It may keep memory between periods in `self`.
    """

    @abstractmethod
    def decide(
        self,
        population: Population,
        proposals: tuple[NDArray[np.int8], ...],
        landscape: Landscape,
        rng: np.random.Generator,
    ) -> "ArrayLike | PopulationDecision":
        """
        The configurations the agents really end up with, one row per agent.

        Return a `PopulationDecision` instead to also say which agents die and which are born;
        births and deaths take effect at the end of the period.

        Parameters
        ----------
        population
            Where every agent was before this period.
        proposals
            What every agent's `step` returned. `proposals[a]` holds the proposals of agent a,
            one per row, best first; `proposals[a][0]` is its first choice. Agents may send
            different numbers of proposals. For an agent that does not exist it is one row of
            -1. Read-only. When several rules are given, a later rule gets what the rule before
            it decided, as one proposal per agent.
        landscape
            The landscape.
        rng
            The rule's own random stream.
        """

    def record(self) -> dict[str, float]:
        """
        Extra values to record in every period, by name. The default records nothing.

        The names must be the same in every period.
        """
        return {}
