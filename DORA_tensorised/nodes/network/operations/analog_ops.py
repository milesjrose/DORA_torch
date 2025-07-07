# nodes/network/operations/analog_ops.py
# Analog operations for Network class

from ...enums import *

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
        self.network = network
    
    def copy(self, analog_number, from_set: Set, to_set: Set):
        """
        Copy an analog from one set to another.

        Args:
            analog (Analog): The analog to copy.
            from_set (Set): The set to copy the analog from.
            to_set (Set): The set to copy the analog to.
        """
        analog_obj = self.network.sets[from_set].get_analog(analog_number)  # get copy of analog
        return self.network.sets[to_set].add_analog(analog_obj)             # add to new set, return the new analog number

    def delete(self, analog, from_set: Set):
        """
        Delete an analog from a set.

        Args:
            analog (int): The analog number.
            from_set (Set): The set to delete the analog from.
        """
        indices = self.network.sets[from_set].token_op.get_analog_indices(analog) # Get indices of tokens in the analog
        self.network.sets[from_set].tensor_op.del_token_indicies(indices)         # Delete the tokens - this will also delete the links to semantics/connections

    def move(self, analog, from_set: Set, to_set: Set):
        """
        Move an analog from one set to another, deleting the old analog.

        Args:
            analog (Analog): The analog to move.
            from_set (Set): The set to move the analog from.
            to_set (Set): The set to move the analog to.
        """
        self.copy(analog, from_set, to_set)
        self.delete(analog, from_set)

    def check_set_match(self):
        """
        Check that the tokens in an analog are from the correct set

        Returns:
            Lists of analog numbers for each set that do not match the set.
            Access with results[set]
        """
        results = []
        for set in Set:
            analogs = self.network.sets[set].token_op.get_analog_where_not(TF.SET, set)
            results.append(analogs)
        return results
    
    def check_for_copy(self):
        """
        Check for analogs in memory that have set != memory.

        Returns:
            List of analog numbers that have set != memory.
        """
        analogs = self.network.sets[Set.MEMORY].token_op.get_analog_where_not(TF.SET, Set.MEMORY)
        return analogs
    
    def clear_set(self, set: Set, analog: int):
        """
        Clear the set feature to "memory" for tokens in an analog.

        Args:
            set (Set): The set of the analog.
            analog (int): The analog number.
        """
        indices = self.network.sets[set].token_op.get_analog_indices(analog)
        self.network.sets[set].token_op.set_features(indices, TF.SET, Set.MEMORY)

    def make_AM_copy(self):
        """
        Copy any analogs with set != memory to AM.
        """
        analogs = self.check_for_copy()
        for analog in analogs:
            analog_obj = self.network.sets[Set.MEMORY].get_analog(analog)           # Get the analog object
            analog_obj.retrieve_lower_tokens()                                      # Set lower tokens to same set as analog
            analog_obj.remove_memory_tokens()                                       # Remove memory tokens
            self.network.sets[analog_obj.set].add_analog(analog_obj)                # Add the analog to the new set

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