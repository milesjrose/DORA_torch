# nodes/network/operations/memory_ops.py
# Memory management operations for Network class

from ...enums import *

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
        self.network = network
    
    
    
    # ---------------------[ TODO: IMPLEMENT ]----------------------------

    def make_am_copy(self):
        """
        Copy any analogues from memory into AM - based on the set feature.
        Equivalent to make_AM_copy() from helpers.py.
        """
        # Implementation using network.sets, network.mappings, etc.
        pass
    
    def check_analog_for_tokens_to_copy(self):
        """
        Find any analogues with set != memory to move.
        """
        # Implementation using network.sets
        pass
    
    def copy_analog(self):
        """
        Copy an analogue from memory to AM.
        """
        # Implementation using network.sets
        pass
    
    def copy_analog_tokens(self):
        """
        Copy each token and add them to the AM set.
        """
        # Implementation using network.sets
        pass
    
    def clear_set(self):
        """
        Sets set feature to memory for tokens in an analogue.
        """
        # Implementation using network.sets
        pass
    
    def delete_unretrieved_tokens(self):
        """
        Delete any tokens from analogue that is unretrieved (set == memory).
        """
        # Implementation using network.sets
        pass
    
    def initialize_am(self):
        """
        Clear activations to all AM.
        """
        # Implementation using network.sets
        pass
    
    def initialize_memory_set(self):
        """
        Clear activation and input to all Memory.
        """
        # Implementation using network.sets
        pass
    
    def clear_all_sets(self):
        """
        Clear the set field of every token in memory (i.e. to clear WM).
        """
        for set in Set:
            if set != Set.MEMORY:
                self.network.sets[set].token_op.set_features_all(TF.SET, Set.MEMORY)
    
    def clear_set(self, set: Set):
        """
        Clear the set of the tokens. (Move to memory set, or delete tokens?)
        """
        self.network.sets[set].token_op.set_features_all(TF.SET, Set.MEMORY)
    
    def reset_inferences(self):
        """
        Reset the inferences of all tokens in memory.
        """
        for set in Set:
            self.network.sets[set].token_op.reset_inferences()
