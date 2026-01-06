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

import logging

# Get the root logger
root_logger = logging.getLogger()


logging.getLogger(__name__).addHandler(logging.NullHandler())

from .network.network import Network
from .network.network_params import Params
from .builder_new import build_network, NetworkBuilder
from .enums import SF as semFields, TF as tokenFields, MappingFields, Type, Set, Mode, OntStatus
from .network.network_params import default_params
from .file_ops import load_network_old, load_network_new, save_network
from .utils.new_printer.printer import Printer
__all__ = [
    "Network",
    "Params",
    "build_network",
    "NetworkBuilder",
    "semFields",
    "tokenFields",
    "MappingFields",
    "Type",
    "Set",
    "Mode",
    "OntStatus",
    "default_params",
    "load_network_old",
    "load_network_new",
    "save_network",
    "Printer"
]

