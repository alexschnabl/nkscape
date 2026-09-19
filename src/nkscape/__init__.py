"""
nkscape: Kauffman NK fitness landscapes and NK-based models of organisations.
"""

from importlib.metadata import version

from nkscape.contribution_tables import ContributionTables
from nkscape.interaction_matrix import InteractionMatrix
from nkscape.landscape import Landscape

__version__ = version("nkscape")

__all__ = ["ContributionTables", "InteractionMatrix", "Landscape", "__version__"]
