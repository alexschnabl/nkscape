"""nkscape: Kauffman NK fitness landscapes and NK-based models of organisations."""

from importlib.metadata import version

from nkscape.interaction_matrix import InteractionMatrix

__version__ = version("nkscape")

__all__ = ["InteractionMatrix", "__version__"]
