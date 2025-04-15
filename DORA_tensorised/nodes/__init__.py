"""
Package: nodes

Provides classes for representing nodes in a network.

Classes:
- Network: Represents the network of nodes.
- Params: Holds shared parameters for the network.

Functions:
- build_network: Builds the network object, takes either a file or a list of symProps. Params can be passed as dict, or added with network.add_params().
- default_params: Returns an object of Params with the default parameters.

node Enums:
- semFields: Semantic fields.
- tokenFields: Token fields.

feature Enums:
- Type: Types of nodes.
- Set: Sets in the network.
- Mode: P token mode.
- OntStatus: Semantic ont status.


"""

from .network.network import Network
from .network.network_params import Params
from .builder.run_build import build_network
from .enums import SF as semFields, TF as tokenFields, MappingFields, Type, Set, Mode, OntStatus
from .network.network_params import default_params

__all__ = [
    "Network",
    "Params",
    "build_network",
    "semFields",
    "tokenFields",
    "MappingFields",
    "Type",
    "Set",
    "Mode",
    "OntStatus",
    "default_params"
]

