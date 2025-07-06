# nodes/network/operations/update_ops.py
# Update operations for Network class

from ...enums import *

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
        self.network = network
    
    # ======================[ ACT FUNCTIONS ]============================

    def initialise_act(self):                                               # Initialise acts in active memory/semantics
        """
        Initialise the acts in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.network.sets[set].initialise_act()

        self.network.semantics.intitialise_sem()
    
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
            self.network.sets[set].initialise_act()
        
        self.network.semantics.initialise_sem()
    
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
        # Implementation using network.semantics
        pass
    
    def del_small_link(self):
        """
        Delete links below threshold.
        """
        # Implementation using network.links
        pass
    
    def round_big_link(self):
        """
        Round links above threshold to 1.
        """
        # Implementation using network.links
        pass 