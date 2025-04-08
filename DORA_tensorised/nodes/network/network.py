# nodes/network/network.py
# Class for holding network sets, and accessing operations on them.

from nodes.enums import *

from .sets import Driver, Recipient, Memory, New_Set, Semantics
from .connections import Mappings
from .network_params import Params

class Network(object):
    """
    A class for holding set objects, and accessing node operations.
    """
    def __init__(self, driver: Driver, recipient: Recipient, LTM: Memory, new_set: New_Set, semantics: Semantics, set_mappings: dict[int, Mappings], params: Params = None):
        """
        Initialize the Nodes object.

        Args:
            driver (Driver): The driver object.
            recipient (Recipient): The recipient object.
            LTM (Tokens): The long-term memory object.
            new_set (Tokens): The new set object. # TODO: check if this needed, or is just a temp set for new tokens
            semantics (Semantics): The semantics object.
            mappings (Mappings): The mappings object.
            DORA_mode (bool): Whether to use DORA mode.
        """
        # node tensors
        self.driver: Driver = driver
        self.recipient: Recipient = recipient
        self.semantics: Semantics = semantics
        self.memory: Memory = LTM
        self.new_set: New_Set = new_set
        self.sets = {
            Set.DRIVER: self.driver,
            Set.RECIPIENT: self.recipient,
            Set.MEMORY: self.memory,
            Set.NEW_SET: self.new_set
        }
        self.params = params
        
        # inter-set connections
        self.set_mappings: Mappings = set_mappings

        # inhibitors
        self.local_inhibitor = 0.0
        self.global_inhibitor = 0.0

        if self.params is not None:
            self.set_params(self.params)
            self.DORA_mode = self.params.DORA_mode
        else:
            self.DORA_mode = True

    def set_params(self, params: Params):                           # Set the params for sets
        """
        Set the parameters for the nodes.
        """
        self.params = params
        self.driver.params = params
        self.recipient.params = params
        self.memory.params = params
        self.new_set.params = params
        self.semantics.params = params
    
    # ======================[ ACT FUNCTIONS ]============================
    def initialise_act(self):                                               # Initialise acts in active memory/semantics
        """
        Initialise the acts in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        self.driver.initialise_act()
        self.recipient.initialise_act()
        self.new_set.initialise_act()
        self.semantics.initialise_sem()

    def update_acts_am(self):                                               # Update acts in active memory/semantics
        """
        Update the acts in the active memory.
        (driver, recipient, new_set, semantics)

        Args:
            gamma (float): Effects the increase in act for each unit.
            delta (float): Effects the decrease in act for each unit.
            hebb_bias (float): The bias for mapping input relative to TD/BU/LATERAL inputs.
        """
        self.driver.update_act()
        self.recipient.update_act()
        self.new_set.update_act()
        self.semantics.update_sem()

    # =======================[ INPUT FUNCTIONS ]=========================
    def initialise_input(self):                                             # Initialise inputs in active memory/semantics
        """
        Initialise the inputs in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        self.driver.initialise_act()
        self.recipient.initialise_act()
        self.new_set.initialise_act()
        self.semantics.initialise_sem()

    def update_inputs_am(self):                                             # TODO: Check if used
        """
        Update the inputs in the active memory.
        (driver, recipient, new_set, semantics)

        Args:
            as_DORA (bool): Whether to use DORA mode.
            phase_set (Int): The current phase set.
            lateral_input_level (float): The lateral input level.
            ignore_object_semantics (bool, optional): Whether to ignore object semantics input. Defaults to False.
        """
        self.driver.update_act()
        self.recipient.update_act()

    # =====================[ INHIBITOR FUNCTIONS ]=======================
    def update_inhibitors(self):                                            # Update inputs and acts of inhibitors
        """
        Update the inputs and acts of the driver and recipient inhibitors.
        (driver, recipient).
        Only updates act of PO if in DORA mode.
        Only updates act of RB in driver.
        """
        #update input driver/recipient RBs, POs
        self.driver.update_inhibitor_input([Type.RB, Type.PO])
        self.recipient.update_inhibitor_input([Type.RB, Type.PO])
        #update driver rb, PO inhibitor act only if in DORA mode
        if self.DORA_mode:
            self.driver.update_inhibitor_act([Type.RB, Type.PO])
            self.recipient.update_inhibitor_act([Type.PO])
        else:
            self.driver.update_inhibitor_act([Type.RB])

    def reset_inhibitors(self):                                             # Reset inhibitors (for RB and PO units) NOTE: Check if required to set for memory and new_set
        """
        Reset the inhibitors (for RB and PO units).
        (driver, recipient, new_set, memory)
        """
        self.driver.reset_inhibitor([Type.RB, Type.PO])
        self.recipient.reset_inhibitor([Type.RB, Type.PO])
        self.memory.reset_inhibitor([Type.RB, Type.PO])
        self.new_set.reset_inhibitor([Type.RB, Type.PO])

    def check_local_inhibitor(self):                                        # Check local inhibition
        """Check local inhibitor activation."""
        if self.driver.check_local_inhibitor():
            self.local_inhibitor = 1.0
    
    def fire_local_inhibitor(self):                                         # Fire local inhibitor
        """Fire the local inhibitor."""
        self.driver.initialise_act(Type.PO)
        self.recipient.initialise_act(Type.PO)
        self.semantics.initialiseSem()
    
    def check_global_inhibitor(self):                                       # Check global inhibition
        """Check global inhibitor activation."""
        if self.driver.check_global_inhibitor():
            self.global_inhibitor = 1.0
        
    def fire_global_inhibitor(self):                                        # Fire global inhibitor
        """Fire the global inhibitor."""
        self.driver.initialise_act([Type.PO, Type.RB, Type.P])
        self.recipient.initialise_act([Type.PO, Type.RB, Type.P])
        self.memory.initialise_act([Type.PO, Type.RB, Type.P])
        self.semantics.initialise_sem()

    # ========================[ NODE FUNCTIONS ]==========================
    def get_pmode_dr(self):                                                 # Get p_mode in driver
        """
        Get parent mode of P units in driver and recipient. Used in time steps activations.
        (driver, recipient)
        """
        self.driver.p_get_mode()
        self.recipient.p_get_mode()
    
    def get_weight_lengths(self):                                           # get weight lenghts in active memory
        """
        Get weight lengths of PO units. Used in run initialisation.
        (driver, recipient, memory, new_set)
        """
        self.driver.po_get_weight_length()
        self.recipient.po_get_weight_length()
        self.memory.po_get_weight_length()
        self.new_set.po_get_weight_length()
    


        
