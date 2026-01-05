# nodes/network/operations/inhibitor_ops.py
# Inhibitor operations for Network class

from ...enums import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..network import Network

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
        self.network: 'Network' = network
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
        self.network.driver().update_op.update_inhibitor_input([Type.RB, Type.PO])
        self.network.recipient().update_op.update_inhibitor_input([Type.RB, Type.PO])
        #update driver rb, PO inhibitor act only if in DORA mode
        if self.network.params.as_DORA:
            self.network.driver().update_op.update_inhibitor_act([Type.RB, Type.PO])
            self.network.recipient().update_op.update_inhibitor_act([Type.PO])
        else:
            self.network.driver().update_op.update_inhibitor_act([Type.RB])

    def reset(self):                                                    # Reset inhibitors (for RB and PO units) NOTE: Check if required to set for memory and new_set
        """
        Reset the inhibitors (for RB and PO units).
        (driver, recipient, new_set, memory)
        """
        self.network.driver().update_op.reset_inhibitor([Type.RB, Type.PO])
        self.network.recipient().update_op.reset_inhibitor([Type.RB, Type.PO])
        self.network.memory().update_op.reset_inhibitor([Type.RB, Type.PO])
        self.network.new_set().update_op.reset_inhibitor([Type.RB, Type.PO])

    def check_local(self):                                              # Check local inhibition
        """Check local inhibitor activation."""
        if self.network.driver().check_local_inhibitor():
            self.local = 1.0
    
    def fire_local(self):                                               # Fire local inhibitor
        """Fire the local inhibitor."""
        self.network.driver().update_op.init_act([Type.PO])
        self.network.recipient().update_op.init_act([Type.PO])
        self.network.semantics.init_sem()
    
    def check_global(self):                                             # Check global inhibition
        """Check global inhibitor activation."""
        if self.network.driver().check_global_inhibitor():
            self.glbal = 1.0
        
    def fire_global(self):                                              # Fire global inhibitor
        """Fire the global inhibitor."""
        self.network.driver().update_op.init_act([Type.PO, Type.RB, Type.P])
        self.network.recipient().update_op.init_act([Type.PO, Type.RB, Type.P])
        self.network.memory().update_op.init_act([Type.PO, Type.RB, Type.P])
        self.network.semantics.init_sem()