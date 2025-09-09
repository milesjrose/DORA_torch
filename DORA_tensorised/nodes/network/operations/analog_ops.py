# nodes/network/operations/analog_ops.py
# Analog operations for Network class

from ...enums import *
from ..single_nodes import Ref_Analog, Analog

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..network import Network

class AnalogOperations:
    """
    Analog operations for the Network class.
    Handles analog management and related functionality.
    """
    
    def __init__(self, network):
        """
        Initialize AnalogOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
    
    def copy(self, analog: Ref_Analog, to_set: Set):
        """
        Copy an analog from one set to another.

        Args:
            analog (Ref_Analog): The analog to copy.
            to_set (Set): The set to copy the analog to.

        Returns:
            Ref_Analog: Reference to the new analog.
        """
        analog_obj: Analog = self.get_analog(analog)       # get copy of analog
        analog_obj.set = to_set                            # set the set of the analog
        return self.add_analog(analog_obj)                 # add to new set, return the new analog reference

    def delete(self, analog: Ref_Analog):
        """
        Delete an analog from a set.

        Args:
            analog (Ref_Analog): The analog to delete.
        """
        indices = self.get_analog_indices(analog)                               # Get indices of tokens in the analog
        self.network.sets[analog.set].tensor_op.del_token_indicies(indices)     # Delete the tokens - this will also delete the links to semantics/connections

    def move(self, analog: Ref_Analog, to_set: Set):
        """
        Move an analog from one set to another, deleting the old analog.

        Args:
            analog (Ref_Analog): The analog to move.
            to_set (Set): The set to move the analog to.
        
        Returns: 
            Ref_Analog: Reference to the new analog.
        """
        new_analog = self.copy(analog, to_set)
        self.delete(analog)
        return new_analog

    def check_set_match(self):
        """
        Check that the tokens in an analog are from the correct set

        Returns:
            Lists of analog references that do not match the set.
        """
        results = []
        for set in Set:
            analogs = self.network.sets[set].token_op.get_analogs_where_not(TF.SET, set)
            results.append(analogs)
        return results
    
    def check_for_copy(self):
        """
        Check for analogs in memory that have set != memory.

        Returns:
            List[Ref_Analog]: References to the analogs that have set != memory.
        """
        analogs = self.network.sets[Set.MEMORY].token_op.get_analogs_where_not(TF.SET, Set.MEMORY)
        return analogs
    
    def clear_set(self, analog: Ref_Analog):
        """
        Clear the set feature to "memory" for tokens in an analog.
        NOTE: Doesn't move the analog to memory set.

        Args:
            analog (Ref_Analog): The analog to clear the set feature for.
        """
        self.set_analog_features(analog, TF.SET, Set.MEMORY)

    def make_AM_copy(self):
        """
        Copy any analogs with set != memory to AM.

        Returns:
            List[Ref_Analog]: References to the analogs that were copied to AM.
        """
        # TODO: Still need to check if should delete from memory set after copy - as think when am set is cleared, ill move then back to memory anyway.
        analogs = self.check_for_copy()
        copied_analogs = []
        for analog in analogs:
            analog_obj = self.get_analog(analog)                  # Get the analog object
            analog_obj.retrieve_lower_tokens()                    # Set lower tokens to same set as analog
            analog_obj.remove_memory_tokens()                     # Remove memory tokens
            new_ref = self.add_analog(analog_obj)                 # Add the analog to the new set
            copied_analogs.append(new_ref)
        return copied_analogs

    def get_analog(self, analog: Ref_Analog):
        """ Get an analog from the network. """
        return self.network.sets[analog.set].get_analog(analog)
    
    def add_analog(self, analog: Analog):
        """
        Add an analog to the network, based on objects set field.

        Args:
            analog (Analog): The analog to add.

        Returns:
            Ref_Analog: Reference to the new analog.
        """
        return self.network.sets[analog.set].add_analog(analog)
    
    def get_analog_indices(self, analog: Ref_Analog):
        """ Get the indices of the tokens in an analog. """
        return self.network.sets[analog.set].token_op.get_analog_indices(analog)
    
    def set_analog_features(self, analog: Ref_Analog, feature: TF, value):
        """ Set a feature of the tokens in an analog. """
        indices = self.get_analog_indices(analog)
        self.network.sets[analog.set].token_op.set_features(indices, feature, value)

    # ---------------------[ TODO: IMPLEMENT ]----------------------------
    
    def find_recip_analog(self):
        """
        Find the analog in the recipient that is mapped to - only used in rel_gen_routine().
        """
        # Implementation using network.sets
        pass
    
    def find_driver_analog_rel_gen(self):
        """
        Find the analog in the driver that maps from - only used in do_rel_gen() in runDORA object.
        """
        # Implementation using network.sets
        pass
    
    def new_set_to_analog(self):
        """
        Put the tokens in the new set into their own analog.
        """
        # Implementation using network.sets
        pass
    
    def print_analog(self):
        """
        Print list of token names in analog.
        """
        # Implementation using network.sets
        pass 