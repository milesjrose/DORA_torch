# nodes.py 
# Class for holding memTypes, and inter-set tensor operations
import torch
from nodeEnums import *
from nodesMemTypes import *
from nodesMemTypes import TF

class Nodes(object):
    def __init__(self, driver: DriverTensor, recipient: RecipientTensor, semantics: TokenTensor, LTM: TokenTensor, links: Links, mappings: Mappings, DORA_mode: bool):
        # node tensors
        self.driver: DriverTensor = driver
        self.recipient: RecipientTensor = recipient
        self.semantics: TokenTensor = semantics
        self.memory: TokenTensor = LTM
        
        # inter-set connections
        self.links: Links = links
        self.mappings: Mappings = mappings
        self.DORA_mode: bool = DORA_mode
    
    # -- UPDATE ACTS --
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
    
    # -- UPDATE INPUTS --
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
    


