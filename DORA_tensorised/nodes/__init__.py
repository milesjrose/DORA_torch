"""
Package: nodes

Provides classes for representing nodes in a network.

- Network: Represents the network of nodes.
- Params: Holds shared parameters for the network.
- build_network: Builds the network object, takes either a file or a list of symProps.
"""

from .network.network import Network
from .network.network_params import Params
from .builder.run_build import build_network

__all__ = [
    "Network",
    "Params",
    "build_network"
]

