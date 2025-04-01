# nodes.py 
# Class for holding memTypes, and inter-set tensor operations
import torch
from nodeEnums import *
from nodeTensors import *

class Nodes(object):
    def __init__(self, driver: DriverTensor, recipient: RecipientTensor, LTM: TokenTensor, new_set: TokenTensor, semantics: SemanticTensor, mappings: Mappings, DORA_mode: bool):
        # node tensors
        self.driver: DriverTensor = driver
        self.recipient: RecipientTensor = recipient
        self.semantics: SemanticTensor = semantics
        self.memory: TokenTensor = LTM
        self.new_set: TokenTensor = new_set
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
        self.update_acts_driver(gamma, delta, hebb_bias)
        self.update_acts_recipient(gamma, delta, hebb_bias)

    # Update acts in driver
    def update_acts_driver(self, gamma, delta, hebb_bias):
        self.driver.update_act(gamma, delta, hebb_bias)

    # Update acts in recipient
    def update_acts_recipient(self, gamma, delta, hebb_bias):
        self.recipient.update_act(gamma, delta, hebb_bias)
    
    # =====================[ UPDATE INPUT FUNCTIONS ]=======================
    # Update inputs in active memory
    def update_inputs_am(self, as_DORA, phase_set, lateral_input_level, ignore_object_semantics=False):
        self.update_inputs_driver(as_DORA)
        self.update_inputs_recpient(as_DORA, phase_set, lateral_input_level, ignore_object_semantics)

    # Update inputs in driver
    def update_inputs_driver(self, as_DORA):
        self.driver.update_act(as_DORA)

    # Update inputs in recipient
    def update_inputs_recpient(self, as_DORA, phase_set, lateral_input_level, ignore_object_semantics=False):
        self.recipient.update_act(as_DORA, phase_set, lateral_input_level, ignore_object_semantics)
    
    # =====================[ INHIBITOR FUNCTIONS ]=======================
    def checkDriverPOs(self):  # Check local inhibitor activation
        if self.driver.check_local_inhibitor():
            self.local_inhibitor = 1.0
    
    def fire_local_inhibitor(self):
        self.driver.initialise_act(Type.PO)
        self.recipient.initialise_act(Type.PO)
        self.semantics.initialiseSem()
    
    def checkDriverRBs(self):   # Check global inhibitor activation
        if self.driver.check_global_inhibitor():
            self.global_inhibitor = 1.0
        
    def fire_global_inhibitor(self):
        self.driver.initialise_act([Type.PO, Type.RB, Type.P])
        self.recipient.initialise_act([Type.PO, Type.RB, Type.P])
        self.memory.initialise_act([Type.PO, Type.RB, Type.P])
        self.semantics.initialise_sem()



