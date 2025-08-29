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
        Initialize mapping hypotheses and connections.
        """
        # Implementation using network.mappings
        pass
    
    def reset_mappings(self):
        """
        Same as reset_mapping_units but also clear max map.
        """
        # Implementation using network.mappings
        pass
    
    
    def update_mapping_hyps(self):
        """
        Update all mapping hypotheses.
        """
        # Implementation using network.mappings
        pass
    
    def reset_mapping_hyps(self):
        """
        Reset the values of mapping hypotheses/max_hyps.
        """
        # Implementation using network.mappings
        pass
    
    def update_mapping_connections(self):
        """
        Update mapping connections.
        """
        # Implementation using network.mappings
        pass
    
    def get_max_map_units(self):
        """
        Get value/token with highest mapping value for each token.
        """
        # Implementation using network.mappings
        pass
    
    def get_max_maps_am(self):
        """
        Get maximum mappings in driver and recipient.
        """
        # Get max maps for driver and recipient
        self.get_max_maps(Set.DRIVER)
        self.get_max_maps(Set.RECIPIENT)
    
    def get_max_maps(self, set: 'Set'):
        """
        Get maximum mappings in set.
        TODO: Check if ever need to check memory mappings in driver?
        """
        if set == Set.DRIVER:
            mappings = self.network.mappings[Set.RECIPIENT]
            nodes = self.network.driver().nodes
            map_weights = mappings[MappingFields.WEIGHT]
            # recipient -> driver mappings, so find max along dim=0
            max_maps = map_weights.max(dim=0).values
            # Set max map for each token
            nodes[:, TF.MAX_MAP] = max_maps
        elif set == Set.RECIPIENT or set == Set.MEMORY:
            mappings = self.network.mappings[set]
            nodes = self.network.sets[set].nodes
            map_weights = mappings[MappingFields.WEIGHT]
            # set -> driver mappings, so find max along dim=1
            max_maps = map_weights.max(dim=1).values
            # Set max map for each token
            nodes[:, TF.MAX_MAP] = max_maps
        
    
    def get_my_max_map(self):
        """
        Return unit that has maximum mapping weight, or null if all mappings == 0.
        """
        # Implementation using network.mappings
        pass
    
    def get_my_max_map_unit(self):
        """
        Same as get_my_max_map.
        """
        # Implementation using network.mappings
        pass 