# nodes/network/routines/routines.py
# Routines object for Network class

from ...enums import *

from typing import TYPE_CHECKING
from .retrieval import RetrievalOperations
from .rel_form import RelFormOperations
from .schematisation import SchematisationOperations
from .rel_gen import RelGenOperations
from .predication import PredicationOperations

if TYPE_CHECKING:
    from ...network import Network

class Routines:
    """
    Routines object for the Network class.
    Handles routines for the network.
    """
    def __init__(self, network):
        """
        Initialize Routines with reference to Network.
        """
        self.network: 'Network' = network
        self.retrieval = RetrievalOperations(self.network)
        self.rel_form = RelFormOperations(self.network)
        self.schematisation = SchematisationOperations(self.network)
        self.rel_gen = RelGenOperations(self.network)
        self.predication = PredicationOperations(self.network)