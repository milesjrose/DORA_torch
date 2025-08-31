# nodes/network/operations/mapping_ops.py
# Mapping operations for Network class

from ...enums import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..network import Network
    from ..sets import Driver, Recipient


class MappingOperations:
    """
    Mapping operations for the Network class.
    Handles mapping hypotheses, connections, and related functionality.
    """
    
    def __init__(self, network):
        """
        Initialize MappingOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network

    # ---------------------[ TODO: IMPLEMENT ]----------------------------
    
    def reset_mapping_units(self):
        """
        Initialize mapping hypotheses and connections in the driver and recipient.
        """
        self.network.mappings[Set.RECIPIENT].reset_mapping_units()
    
    def reset_mappings(self):
        """
        Initialise mapping hypotheses, connections, and max map for all tokens.
        """
        self.network.mappings[Set.MEMORY].reset_mappings()
        self.network.mappings[Set.RECIPIENT].reset_mappings()

    
    def update_mapping_hyps(self):
        """
        Update all mapping hypotheses.
        """
        self.network.mappings[Set.RECIPIENT].update_hypotheses()
        # TODO: Check if we update memory set hypotheses here?
    
    def reset_mapping_hyps(self):
        """
        Reset the values of mapping hypotheses/max_hyps.
        """
        self.network.mappings[Set.RECIPIENT].reset_hypotheses()
        # TODO: Check reset mem set here as well?
    
    def update_mapping_connections(self):
        """
        Update mapping connections.
        """
        self.network.mappings[Set.RECIPIENT].update_connections(self.network.params.eta)
    
    def get_max_maps(self):
        """
        Get value/token with highest mapping value for each token in driver and recipient.
        """
        max_recipient, max_driver = self.network.mappings[Set.RECIPIENT].get_max_map()
        # Set max map for driver
        self.network.sets[Set.DRIVER].nodes[:, TF.MAX_MAP] = max_driver.values
        self.network.sets[Set.DRIVER].nodes[:, TF.MAX_MAP_UNIT] = max_driver.indices
        # Set max map for recipient
        self.network.sets[Set.RECIPIENT].nodes[:, TF.MAX_MAP] = max_recipient.values
        self.network.sets[Set.RECIPIENT].nodes[:, TF.MAX_MAP_UNIT] = max_recipient.indices
    
    def get_max_map_memory(self):
        """
        Get value/token with highest mapping value for each token in memory.
        TODO: Check if this is ever needed?
        """
        max_memory, max_driver = self.network.mappings[Set.MEMORY].get_max_map()
        self.network.sets[Set.MEMORY].nodes[:, TF.MAX_MAP] = max_memory.values
        self.network.sets[Set.MEMORY].nodes[:, TF.MAX_MAP_UNIT] = max_memory.indices