# nodes/network/operations/mapping_ops.py
# Mapping operations for Network class

from ...enums import *
from typing import TYPE_CHECKING

from ..single_nodes import Ref_Token

if TYPE_CHECKING:
    from ..network import Network


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
    
    def reset_mapping_units(self):
        """
        Initialize mapping hypotheses and connections in the driver and recipient.
        """
        self.network.mappings.reset_mapping_units()
    
    def reset_mappings(self):
        """
        Initialise mapping hypotheses, connections, and max map for all tokens.
        """
        self.network.mappings.reset_mappings()

    def update_mapping_hyps(self):
        """
        Update all mapping hypotheses.
        """
        self.network.mappings.update_hypotheses()
    
    def reset_mapping_hyps(self):
        """
        Reset the values of mapping hypotheses/max_hyps.
        """
        self.network.mappings.reset_hypotheses()
    
    def update_mapping_connections(self):
        """
        Update mapping connections.
        """
        self.network.mappings.update_weight(self.network.params.eta)
    
    def get_max_maps(self, set: list[Set] = [Set.RECIPIENT, Set.DRIVER]):
        """
        Get value/token with highest mapping value for each token in sets.
        
        Args:
            set (Set, optional): The set to get max maps for. Defaults to [Set.RECIPIENT, Set.DRIVER].
        """
        max_recipient, max_driver = self.network.mappings.get_max_map()
        # Set max map for driver
        if Set.DRIVER in set:
            self.network.sets[Set.DRIVER].token_op.set_max_maps(max_driver.values)
            self.network.sets[Set.DRIVER].token_op.set_max_map_units(max_driver.indices)
        # Set max map for recipient
        if Set.RECIPIENT in set:
            self.network.sets[Set.RECIPIENT].token_op.set_max_maps(max_recipient.values)
            self.network.sets[Set.RECIPIENT].token_op.set_max_map_units(max_recipient.indices)
    
    def get_max_map_unit(self, idx: int) -> Ref_Token:
        """ Get a reference to the unit that the token maps to most """
        mapped_idx = self.network.token_tensor.get_feature(idx, TF.MAX_MAP_UNIT)
        return int(mapped_idx.item())
