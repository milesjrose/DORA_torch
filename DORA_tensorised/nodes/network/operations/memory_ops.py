# nodes/network/operations/memory_ops.py
# Memory management operations for Network class

from ...enums import *
from ..sets import Driver, Recipient
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..network import Network
    from ..single_nodes import Ref_Analog

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
    
    # --------[ NOTE: I think i ported everything. Idk where these functions went though. ]--------

    def make_am(self, copy = True):
        """
        Copy any analogues from memory into AM - based on the set feature.
        Equivalent to make_AM_copy() from helpers.py.
        """
        analogs = self.network.analog.check_for_copy()
        for analog in analogs:
            if copy:
                self.network.analog.copy(analog, Set.DRIVER)
            else:
                self.network.analog.move(analog, Set.DRIVER)
    
    def check_analog_for_tokens_to_copy(self) -> list[Ref_Analog]:
        """Find any analogues with set != memory to move."""
        analogs = self.network.analog.check_for_copy()
        return analogs

    def del_mem_tokens(self, sets: list[Set] = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]):
        """ delete any memory tokens from the am """
        for set in sets:
            if set not in [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]:
                raise ValueError(f"Set {set} is not a valid set for del_mem_tokens")
            self.network.sets[set].tensor_op.del_mem_tokens()

    def clear_all_sets(self):
        """Clear the set field of every token in memory (i.e. to clear WM)."""
        for set in Set:
            if set != Set.MEMORY:
                self.network.sets[set].token_op.set_features_all(TF.SET, Set.MEMORY)
    
    def clear_set(self, set: Set):
        """Clear the set of the tokens. (Move to memory set, or delete tokens?)"""
        self.network.sets[set].token_op.set_features_all(TF.SET, Set.MEMORY)
        # if the tokens are retrieved from memory, delete them, otherwise move them to memory I think?
        # this gets painful if any retrieved tokens are connected to other tokens in the set.
    
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
        # Create new driver and recipient objects by passsing in opposites tensors
        old = self.network.recipient()
        new_nodes = old.nodes.clone()
        new_nodes[:, TF.SET] = Set.DRIVER
        new_driver = Driver(new_nodes, old.connections, old.IDs, old.names)
        old = self.network.driver()
        new_nodes = old.nodes.clone()
        new_nodes[:, TF.SET] = Set.RECIPIENT
        new_rec = Recipient(new_nodes, old.connections, old.IDs, old.names)
        self.network.sets[Set.DRIVER] = new_driver
        self.network.sets[Set.RECIPIENT] = new_rec
        # Update mappings
        # Just transpose the mapping tensor - idk if we should clear mappings here?
        self.network.mappings[Set.RECIPIENT].swap_driver_recipient()
        # Update links
        self.network.links.swap_driver_recipient()

        
