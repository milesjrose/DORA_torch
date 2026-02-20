# nodes/network/operations/update_ops.py
# Update operations for Network class

from typing import TYPE_CHECKING
from ...enums import *

if TYPE_CHECKING:
    from ..network import Network

class UpdateOperations:
    """
    Update operations for the Network class.
    Handles input and activation updates across sets.
    """
    
    def __init__(self, network):
        """
        Initialize UpdateOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
    
    # ======================[ ACT FUNCTIONS ]============================
    # NOTE: Now that we have a single tensor for everything, anything that loops through 
    #       sets should be moved to a single operation as no need to do seperately.


    def initialise_act(self, no_ret: bool = False):
        """
        Initialise the act and inputs in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        
        Args:
            no_ret (bool): If true, don't initialise retrieved tokens in the recipient. (For do_retrieval)
        """
        sets = [Set.DRIVER, Set.NEW_SET]
        # If not initialising retrieved recipient tokens, do it seperately. O.w do in other loop.
        if no_ret:
            self.network.recipient().update_op.init_act_ignore_retrieved([Type.GROUP, Type.P, Type.RB, Type.PO])
        else:
            sets.append(Set.RECIPIENT)
        # Initialise sets
        for set in sets:
            self.network.sets[set].update_op.init_act([Type.GROUP, Type.P, Type.RB, Type.PO])

        self.network.semantics.init_sem()
    
    def initialise_act_memory(self):
        """
        Initialise the acts and inputs in the memory.
        (memory)
        """
        self.network.sets[Set.MEMORY].update_op.init_act([Type.GROUP, Type.P, Type.RB, Type.PO])

    def acts(self, set: Set): 
        """
        Update the acts in the given set.

        Args:
            set (Set): The set to update acts in.
        """
        self.network.sets[set].update_op.update_act()
    
    def acts_sem(self):
        """
        Update the acts in the semantics.
        """
        self.network.semantics.update_act()

    def acts_am(self): 
        """
        Update the acts in the active memory.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.acts(set)
        
        self.acts_sem()
    
    # =======================[ INPUT FUNCTIONS ]=========================

    def initialise_input(self): 
        """
        Initialise the inputs in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.network.sets[set].update_op.init_input([Type.GROUP, Type.P, Type.RB, Type.PO], 0.0)
        
        self.network.semantics.init_input(0.0)

    def initialise_input_memory(self):
        """
        Initialise the inputs in the memory.
        (memory)
        """
        self.network.sets[Set.MEMORY].update_op.init_input([Type.GROUP, Type.P, Type.RB, Type.PO], 0.0)
    
    def inputs(self, set: Set, ignore_modes: bool = False):
        """
        Update the inputs in the given token set.

        Args:
            set (Set): The set to update inputs in.
            ignore_modes (bool, optional): Whether to ignore p modes (updating all P as parents, for retrieval) for memory set. Defaults to False.
        """
        if set == Set.DRIVER or set == Set.NEW_SET:
            self.network.sets[set].update_input()
        elif set == Set.RECIPIENT:
            self.network.sets[set].update_input(self.network.semantics, self.network.links)
        elif set == Set.MEMORY:
            self.network.memory().update_input(self.network.semantics, self.network.links, ignore_modes)
    
    def inputs_sem(self):               
        """
        Update the inputs in the semantics.
        """
        self.network.semantics.update_input(self.network.sets[Set.DRIVER], self.network.sets[Set.RECIPIENT])

    def inputs_am(self):
        """
        Update the inputs in the active memory.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.inputs(set)
        
        self.inputs_sem()

    # ======================[ SEM/LINK FUNCTIONS]=======================

    def get_max_sem_input(self) -> float:
        """
        Get maximum semantic input.

        Returns:
            float: The maximum semantic input.
            int: The index of the semantic with the maximum input.
        """
        return self.network.semantics.get_max_input()
    
    
    def set_max_sem_input(self, max_input: float):
        """
        Set maximum semantic input.

        Args:
            max_input (float): The maximum semantic input.
        """
        self.network.semantics.set_max_input(max_input)
    
    def max_sem_input(self):
        """ Update the max_sem_input field of all semantics."""
        self.network.semantics.update_max_inputs()
    
    def del_small_link(self, threshold: float):
        """
        Delete links below threshold.
        """
        self.network.links.del_small_link(threshold)
    
    def round_big_link(self, threshold: float):
        """
        Round links above threshold to 1.
        """
        self.network.links.round_big_link(threshold)