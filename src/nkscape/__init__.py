"""
nkscape: Kauffman NK fitness landscapes and NK-based models of organisations.
"""

from importlib.metadata import version

from nkscape.agent import Agent
from nkscape.contribution_tables import ContributionTables
from nkscape.interaction_matrix import InteractionMatrix
from nkscape.landscape import Landscape, get_enumeration_limit, set_enumeration_limit
from nkscape.simulation import RunResult, run

__version__ = version("nkscape")

__all__ = [
    "Agent",
    "ContributionTables",
    "InteractionMatrix",
    "Landscape",
    "RunResult",
    "__version__",
    "get_enumeration_limit",
    "run",
    "set_enumeration_limit",
]
