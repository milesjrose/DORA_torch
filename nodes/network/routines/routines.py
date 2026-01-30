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
    Provides a unified interface to the learning routines for the Network:
    - retrieval: Retrieves co-active structures in memory to the recipient.
    - predication: Learns predicates from object through shared features.
    - rel_form: Learns multiplace relational structures by linking co-occuring RB pairs.
    - rel_gen: Infer structure in the recipient based on unmapped structure in the driver.
    - schematisation: Infer structure into the newSet based on the driver and recipient.
    """
    def __init__(self, network):
        """
        Initialize Routines with reference to Network.
        """
        self.network: 'Network' = network
        self.retrieval: RetrievalOperations = RetrievalOperations(self.network)
        self.rel_form: RelFormOperations = RelFormOperations(self.network)
        self.rel_gen: RelGenOperations = RelGenOperations(self.network)
        self.schema: SchematisationOperations = SchematisationOperations(self.network)
        self.predication: PredicationOperations = PredicationOperations(self.network)