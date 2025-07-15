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

    def initialise_act(self):                                               # Initialise acts in active memory/semantics
        """
        Initialise the acts in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.network.sets[set].init_act([Type.GROUP, Type.P, Type.RB, Type.PO])

        self.network.semantics.init_sem()
    
    def initialise_act_memory(self):
        """
        Initialise the acts in the memory.
        (memory)
        """
        self.network.memory.init_act()

    def acts(self, set: Set):                                        # Update acts in given token set    
        """
        Update the acts in the given set.

        Args:
            set (Set): The set to update acts in.
        """
        self.network.sets[set].update_act()
    
    def acts_sem(self):                                              # Update acts in semantics
        """
        Update the acts in the semantics.
        """
        self.network.semantics.update_act()

    def acts_am(self):                                               # Update acts in active memory/semantics
        """
        Update the acts in the active memory.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.acts(set)
        
        self.acts_sem()
    
    # =======================[ INPUT FUNCTIONS ]=========================

    def initialise_input(self):                                             # Initialise inputs in active memory/semantics
        """
        Initialise the inputs in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.network.sets[set].init_input([Type.GROUP, Type.P, Type.RB, Type.PO], 0.0)
        
        self.network.semantics.init_input(0.0)

    def initialise_input_memory(self):
        """
        Initialise the inputs in the memory.
        (memory)
        """
        self.network.memory.init_input(0.0)
    
    def inputs(self, set: Set):                                      # Update inputs in given token set
        """
        Update the inputs in the given token set.

        Args:
            set (Set): The set to update inputs in.
        """
        self.network.sets[set].update_input()
    
    def inputs_sem(self):                                            # Update inputs in semantics               
        """
        Update the inputs in the semantics.
        """
        self.network.semantics.update_input(self.network.sets[Set.DRIVER], self.network.sets[Set.RECIPIENT])

    def inputs_am(self):                                             # Update inputs in active memory
        """
        Update the inputs in the active memory.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.inputs(set)
        
        self.inputs_sem()

    # ======================[ TODO: IMPLEMENT ]=======================
    def get_max_sem_input(self):
        """
        Get maximum semantic input.
        """
        return self.network.semantics.get_max_input()
    
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