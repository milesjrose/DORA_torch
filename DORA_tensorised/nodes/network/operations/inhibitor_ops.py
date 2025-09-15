# nodes/network/operations/inhibitor_ops.py
# Inhibitor operations for Network class

from ...enums import *

class InhibitorOperations:
    """
    Inhibitor operations for the Network class.
    Handles inhibitor management.
    """
    
    def __init__(self, network):
        """
        Initialize InhibitorOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network = network
        self.sets = network.sets
        # NOTE: Not sure where/if these are used? 
        self.local = 0.0
        self.glbal = 0.0

    def update(self):                                                   # Update inputs and acts of inhibitors
        """
        Update the inputs and acts of the driver and recipient inhibitors.
        (driver, recipient).
        - Only updates act of PO if in DORA mode.
        - Only updates act of RB in driver.
        """
        #update input driver/recipient RBs, POs
        self.sets[Set.DRIVER].update_inhibitor_input([Type.RB, Type.PO])
        self.sets[Set.RECIPIENT].update_inhibitor_input([Type.RB, Type.PO])
        #update driver rb, PO inhibitor act only if in DORA mode
        if self.DORA_mode:
            self.sets[Set.DRIVER].update_inhibitor_act([Type.RB, Type.PO])
            self.sets[Set.RECIPIENT].update_inhibitor_act([Type.PO])
        else:
            self.sets[Set.DRIVER].update_inhibitor_act([Type.RB])

    def reset(self):                                                    # Reset inhibitors (for RB and PO units) NOTE: Check if required to set for memory and new_set
        """
        Reset the inhibitors (for RB and PO units).
        (driver, recipient, new_set, memory)
        """
        self.sets[Set.DRIVER].reset_inhibitor([Type.RB, Type.PO])
        self.sets[Set.RECIPIENT].reset_inhibitor([Type.RB, Type.PO])
        self.sets[Set.MEMORY].reset_inhibitor([Type.RB, Type.PO])
        self.sets[Set.NEW_SET].reset_inhibitor([Type.RB, Type.PO])

    def check_local(self):                                              # Check local inhibition
        """Check local inhibitor activation."""
        if self.sets[Set.DRIVER].check_local_inhibitor():
            self.local = 1.0
    
    def fire_local(self):                                               # Fire local inhibitor
        """Fire the local inhibitor."""
        self.sets[Set.DRIVER].initialise_act(Type.PO)
        self.sets[Set.RECIPIENT].initialise_act(Type.PO)
        self.semantics.initialiseSem()
    
    def check_global(self):                                             # Check global inhibition
        """Check global inhibitor activation."""
        if self.sets[Set.DRIVER].check_global_inhibitor():
            self.glbal = 1.0
        
    def fire_global(self):                                              # Fire global inhibitor
        """Fire the global inhibitor."""
        self.sets[Set.DRIVER].initialise_act([Type.PO, Type.RB, Type.P])
        self.sets[Set.RECIPIENT].initialise_act([Type.PO, Type.RB, Type.P])
        self.sets[Set.MEMORY].initialise_act([Type.PO, Type.RB, Type.P])
        self.semantics.initialise_sem()