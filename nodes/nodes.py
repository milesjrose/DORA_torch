# nodes.py 
# Class for holding memTypes, and inter-set tensor operations
import torch
from .nodeEnums import *
from .nodeTensors import *

class Nodes(object):
    """
    A class for holding token tensors for each set, and accessing node operations.
    """
    def __init__(self, driver: Driver, recipient: Recipient, LTM: Tokens, new_set: Tokens, semantics: Semantic, mappings: Mappings, DORA_mode: bool):
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
        self.semantics: Semantic = semantics
        self.memory: Tokens = LTM
        self.new_set: Tokens = new_set
        self.sets = {
            Set.DRIVER: self.driver,
            Set.RECIPIENT: self.recipient,
            Set.MEMORY: self.memory,
            Set.NEW_SET: self.new_set
        }
        
        # inter-set connections
        self.mappings: Mappings = mappings
        self.DORA_mode: bool = DORA_mode

        # inhibitors
        self.local_inhibitor = 0.0
        self.global_inhibitor = 0.0
    
    # =====================[ UPDATE ACT FUNCTIONS ]=======================
    # Update of acts in active memory
    def update_acts_am(self, gamma, delta, hebb_bias):
        """Update the acts in the active memory."""
        self.update_acts_driver(gamma, delta, hebb_bias)
        self.update_acts_recipient(gamma, delta, hebb_bias)

    # Update acts in driver
    def update_acts_driver(self, gamma, delta, hebb_bias):
        """Update the acts in the driver."""
        self.driver.update_act(gamma, delta, hebb_bias)

    # Update acts in recipient
    def update_acts_recipient(self, gamma, delta, hebb_bias):
        """Update the acts in the recipient."""
        self.recipient.update_act(gamma, delta, hebb_bias)
    
    # =====================[ UPDATE INPUT FUNCTIONS ]=======================
    # Update inputs in active memory
    def update_inputs_am(self, as_DORA, phase_set, lateral_input_level, ignore_object_semantics=False):
        """
        Update the inputs in the active memory.
        
        Args:
            as_DORA (bool): Whether to use DORA mode.
            phase_set (Int): The current phase set.
            lateral_input_level (float): The lateral input level.
            ignore_object_semantics (bool, optional): Whether to ignore object semantics input. Defaults to False.
        """
        self.update_inputs_driver(as_DORA)
        self.update_inputs_recpient(as_DORA, phase_set, lateral_input_level, ignore_object_semantics)

    # Update inputs in driver
    def update_inputs_driver(self, as_DORA):
        """
        Update the inputs in the driver.
        
        Args:
            as_DORA (bool): Whether to use DORA mode.
        """
        self.driver.update_act(as_DORA)

    # Update inputs in recipient
    def update_inputs_recpient(self, as_DORA, phase_set, lateral_input_level, ignore_object_semantics=False):
        """
        Update the inputs in the recipient.
        
        Args:
            as_DORA (bool): Whether to use DORA mode.
            phase_set (int): The current phase set.
            lateral_input_level (float): The lateral input level.
            ignore_object_semantics (bool, optional): Whether to ignore object semantics input. Defaults to False.
        """
        self.recipient.update_act(as_DORA, phase_set, lateral_input_level, ignore_object_semantics)
    
    # =====================[ INHIBITOR FUNCTIONS ]=======================
    def checkDriverPOs(self):
        """Check local inhibitor activation."""
        if self.driver.check_local_inhibitor():
            self.local_inhibitor = 1.0
    
    def fire_local_inhibitor(self):
        """Fire the local inhibitor."""
        self.driver.initialise_act(Type.PO)
        self.recipient.initialise_act(Type.PO)
        self.semantics.initialiseSem()
    
    def checkDriverRBs(self):
        """Check global inhibitor activation."""
        if self.driver.check_global_inhibitor():
            self.global_inhibitor = 1.0
        
    def fire_global_inhibitor(self):
        """Fire the global inhibitor."""
        self.driver.initialise_act([Type.PO, Type.RB, Type.P])
        self.recipient.initialise_act([Type.PO, Type.RB, Type.P])
        self.memory.initialise_act([Type.PO, Type.RB, Type.P])
        self.semantics.initialise_sem()



