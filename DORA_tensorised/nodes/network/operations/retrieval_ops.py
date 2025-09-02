# nodes/network/operations/retrieval_ops.py
# Retrieval operations for Network class

from ...enums import *
import torch
from math import exp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...network import Network
    from ..sets import Driver, Recipient, Memory

class RetrievalOperations:
    """
    Retrieval operations for the Network class.
    Handles token retrieval and related functionality.
    """
    
    def __init__(self, network):
        """
        Initialize RetrievalOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network

    # ---------------------[ TODO: IMPLEMENT ]----------------------------
    
    def retrieval_routine(self):
        """
        Run the model retrieval routine - update input/act in memory, then if bias_retrieval_analogs 
        get the total act for each analog, else track the most active tokens in memory.
        """
        memory: 'Memory' = self.network.memory()
        memory.update_input()
        memory.update_act()
        if self.network.params.bias_retrieval_analogs:
            memory.tensor_ops.get_analog_activation_counts_scatter()
        else:
            memory.token_ops.get_max_acts()

    
    def retrieve_tokens(self):
        """
        Retrieve tokens from memory.
        """
        use_rel_act = self.network.params.use_relative_act
        analog_bias = self.network.params.bias_retrieval_analogs
        # 2. calc normal act for each analog
        analogs = self.network.memory().analogs
        counts = self.network.memory().analog_counts
        acts = self.network.memory().analog_activations
        # Total act/count for each
        normal_act = acts/counts
        # take weighted average of normal_acts
        avg_norm =  (torch.mean(normal_act) + torch.max(normal_act))/2
        # transform the norm_acts with
        # 1 / (1 + exp(10 * norm_act - avg_norm))
        normal_act = 1 / (1 + exp(10 * normal_act - avg_norm))
        sum_normal_act = torch.sum(normal_act)
        # Retrieve 2
    
    def retrieve_analog_contents(self):
        """
        Set all tokens in analog to recipient.
        """
        # Implementation using network.sets
        pass
    
    def get_most_active_unit(self):
        """
        Take in a set of nodes (e.g. all POs or Recipient RBs) and return the most active unit.
        Make sure activation != 0.0 (i.e. an active unit not just the first if no units active).
        """
        # Implementation using network.sets
        pass
    
    def get_most_active_punit(self):
        """
        Basically just get_most_active_unit but specifically for POs in a given mode.
        """
        # Implementation using network.sets
        pass 