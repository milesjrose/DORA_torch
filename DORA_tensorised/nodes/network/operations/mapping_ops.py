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
    
    def reset_mapping_hyps(self):
        """
        Reset the values of mapping hypotheses/max_hyps.
        """
        self.network.mappings[Set.RECIPIENT].reset_hypotheses()
    
    def update_mapping_connections(self):
        """
        Update mapping connections.
        """
        self.network.mappings[Set.RECIPIENT].update_connections(self.network.params.eta)
    
    def get_max_maps(self, set: list[Set] = [Set.RECIPIENT, Set.DRIVER]):
        """
        Get value/token with highest mapping value for each token in sets.
        
        Args:
            set (Set, optional): The set to get max maps for. Defaults to [Set.RECIPIENT, Set.DRIVER].
        """
        max_recipient, max_driver = self.network.mappings[Set.RECIPIENT].get_max_map()
        # Set max map for driver
        if Set.DRIVER in set:
            self.network.sets[Set.DRIVER].nodes[:, TF.MAX_MAP] = max_driver.values
            self.network.sets[Set.DRIVER].nodes[:, TF.MAX_MAP_UNIT] = max_driver.indices
        # Set max map for recipient
        if Set.RECIPIENT in set:
            self.network.sets[Set.RECIPIENT].nodes[:, TF.MAX_MAP] = max_recipient.values
            self.network.sets[Set.RECIPIENT].nodes[:, TF.MAX_MAP_UNIT] = max_recipient.indices