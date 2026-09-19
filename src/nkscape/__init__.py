"""
nkscape: Kauffman NK fitness landscapes and NK-based models of organisations.
"""

from importlib.metadata import version

from nkscape.contribution_tables import ContributionTables
from nkscape.interaction_matrix import InteractionMatrix

__version__ = version("nkscape")

__all__ = ["ContributionTables", "InteractionMatrix", "__version__"]
