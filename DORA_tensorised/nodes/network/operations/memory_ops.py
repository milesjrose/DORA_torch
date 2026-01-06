# nodes/network/operations/memory_ops.py
# Memory management operations for Network class

from ...enums import *
from ..sets import Driver, Recipient
from ..single_nodes import Ref_Analog
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..network import Network

class TensorOperations:
    """
    Memory management and memory wide set operations for the Network class.
    Handles copying, clearing, and managing memory sets.
    """
    
    def __init__(self, network):
        """
        Initialize MemoryOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
   
    def make_am(self, copy = True):
        """
        Copy any analogues from memory into AM - based on the set feature.
        Equivalent to make_AM_copy() from helpers.py.
        """
        raise NotImplementedError("Function is moved to analog_ops.py")
    
    def check_analog_for_tokens_to_copy(self) -> list[Ref_Analog]:
        """Find any analogues with set != memory to move."""
        raise NotImplementedError("Function is moved to analog_ops.py")

    def del_mem_tokens(self, sets: list[Set] = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]):
        """ delete any memory tokens from the am.
        TODO: Check usage -> there shouldn't be any memory tokens in am after make_am, """
        raise NotImplementedError("Function is not used anymore")
        #for set in sets:
        #       if set not in [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]:
        #        raise ValueError(f"Set {set} is not a valid set for del_mem_tokens")
        #    self.network.sets[set].tensor_op.del_mem_tokens()

    def clear_all_sets(self):
        """Clear the set field of every token in memory (i.e. to clear WM)."""
        all_idxs = self.network.token_tensor.cache.get_all_nodes_indices()
        self.network.token_tensor.set_feature(all_idxs, TF.SET, Set.MEMORY)
        self.network.recache()
    
    def clear_set(self, set: Set):
        """Clear the set of the tokens. (Move to memory set, or delete tokens?)"""
        self.network.sets[set].token_op.set_features_all(TF.SET, Set.MEMORY)
        self.network.recache()
    
    def reset_inferences(self):
        """Reset the inferences of all tokens."""
        for set in Set:
            self.network.sets[set].token_op.reset_inferences()
    
    def reset_maker_made_units(self):
        """Reset the maker and made units of all tokens."""
        for set in Set:
            self.network.sets[set].token_op.reset_maker_made_units()
    
    def swap_driver_recipient(self):
        """Swap the contents of the driver and recipient"""
        # set all driver tokens to recipient set, and vice versa
        driver_idxs = self.network.token_tensor.cache.get_set_indices(Set.DRIVER)
        recipient_idxs = self.network.token_tensor.cache.get_set_indices(Set.RECIPIENT)
        self.network.token_tensor.set_feature(driver_idxs, TF.SET, Set.RECIPIENT)
        self.network.token_tensor.set_feature(recipient_idxs, TF.SET, Set.DRIVER)
        # update mappings
        self.network.mappings.swap_driver_recipient()
        # recache
        self.network.recache()

        
