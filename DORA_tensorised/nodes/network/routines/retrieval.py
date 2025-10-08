# nodes/network/operations/retrieval_ops.py
# Retrieval operations for Network class

from ...enums import *
import torch
from math import exp
from logging import getLogger
from ..single_nodes import Ref_Analog
from typing import TYPE_CHECKING
from random import random
from ...utils import tensor_ops as tOps

logger = getLogger(__name__)

if TYPE_CHECKING:
    from ...network import Network
    from ..sets import Driver, Recipient, Memory

class RetrievalOperations:
    """
    Retrieval operations for the Network class.
    Handles token retrieval and related functionality.
    """

    # TODO: finish token retrieval, and add tests.
    
    def __init__(self, network):
        """
        Initialize RetrievalOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
        self.debug = False
    
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
        Biases retrieval to analogs and uses relative act based on network params.
        """
        use_rel_act = self.network.params.use_relative_act
        analog_bias = self.network.params.bias_retrieval_analogs
        if analog_bias:
            self.retrieve_analogs_biased(use_rel_act)
        else:
            self.retrieve_tokens_no_bias(use_rel_act)

    def retrieve_analogs_biased(self, use_relative_act):
        """ Retrieve analogs from memory, using analog bias"""
        self.network.memory().tensor_ops.get_analog_activation_counts_scatter()
        # Calc normal act for each analog
        analogs = self.network.memory().analogs
        counts = self.network.memory().analog_counts
        acts = self.network.memory().analog_activations
        normal_act = acts/counts
        
        # If relative act, transform normal_act with sigmoidal function
        if use_relative_act:
            # take weighted average of normal_acts
            avg_norm =  (torch.mean(normal_act) + torch.max(normal_act))/2
            # transform the norm_acts with
            # 1 / (1 + exp(10 * (norm_act - avg_norm)))
            normal_act = 1 / (1 + torch.exp(10 * (normal_act - avg_norm)))
            sum_normal_act = torch.sum(normal_act)
        else:
            # For non-relative activation, use normal_act directly
            sum_normal_act = torch.sum(normal_act)

        # Retrieve analogs with luce choice
        active_mask = counts > 0                    # Mask 0 act analogs
        if active_mask.sum() > 0 and sum_normal_act > 0:
            # Calc retrieval prob for each analog
            retrieve_prob = normal_act[active_mask]/sum_normal_act
            random_num = torch.rand(analogs.shape[0])
            retrieve_mask = active_mask & (retrieve_prob > random_num)
            # Retrieve analogs NOTE: not vectorised TODO: Add method for moving multiple analogs at once
            for analog in analogs[retrieve_mask]:
                self.retrieve_analog(analog)
        
    def retrieve_tokens_no_bias(self, use_relative_act):
        """Retrieve tokens from memory, no bias to analogs"""
        if use_relative_act:
            raise NotImplementedError("Relative act not implemented for non-bias retrieval")

        # Update max acts, and get all mask
        self.network.memory().token_ops.get_max_acts()                     
        all_mask = self.network.memory().tensor_op.get_all_nodes_mask()

        # Decide on retrieval based on luce choice
        def luce_choice_retrieval(token_sum, token_mask):
            # make sure token_sum > 0
            if token_sum <= 0:
                return torch.zeros_like(token_mask, dtype=torch.bool)
            # retrieve prob = max_act / token_sum
            retrieve_prob = self.network.memory().nodes[token_mask, TF.MAX_ACT] / token_sum
            # if retrieve prob > random num, flag token for retrieval
            random_num = torch.rand_like(retrieve_prob)
            # Create mask for tokens that should be retrieved
            retrieve_mask = torch.zeros_like(token_mask, dtype=torch.bool)
            retrieve_mask[token_mask] = retrieve_prob > random_num
            return retrieve_mask
        
        # Apply luce choice to each token type
        retrieve_mask = torch.zeros_like(all_mask, dtype=torch.bool)
        for token_type in [Type.P, Type.RB, Type.PO]:
            token_mask = self.network.memory().tensor_op.get_mask(token_type)
            token_sum = self.network.memory().nodes[token_mask, TF.MAX_ACT].sum()
            type_retrieve_mask = luce_choice_retrieval(token_sum, token_mask)
            retrieve_mask = retrieve_mask | type_retrieve_mask

        # Move tokens to recipient
        self.retrieve_tokens_with_mask(retrieve_mask)

            
    def retrieve_analog(self, analog: int):
        """
        Move analog from memory to recipient
        """
        ref = Ref_Analog(analog, Set.MEMORY)
        self.network.analog.move(ref, Set.RECIPIENT)
        # Set retrieved to true
        self.network.analog.set_analog_features(ref, TF.RETRIEVED, B.TRUE)
    
    def retrieve_tokens_with_mask(self, token_mask: torch.Tensor):
        """
        Move tokens in mask from memory to recipient, including any children of these tokens.
        """
        if not token_mask.any():
            return
        memory = self.network.memory()
        # Get indices of tokens to retrieve
        highest_token = torch.max(memory.nodes[token_mask, TF.TYPE]).item()
        levels_of_children = int(highest_token - 1)
        # Get children
        for i in range(levels_of_children):
            logger.debug(f"mask shape: {token_mask.shape}")
            children = memory.token_op.get_children(token_mask)
            logger.debug(f"children shape: {children.shape}")
            token_mask |= children
        
        # for now just move them all to a new analog, then move the analog
        original_analogs = memory.nodes[token_mask, TF.ANALOG]
        #set new
        token_indices = torch.where(token_mask)[0]
        new_analog_id = memory.tensor_op.get_new_analog_id()
        memory.token_op.set_features(token_indices, TF.ANALOG, new_analog_id)
        # move the analog to recipient
        new_analog = self.network.analog.move(Ref_Analog(new_analog_id, Set.MEMORY), Set.RECIPIENT)
        # set retrieved to true
        self.network.analog.set_analog_features(new_analog, TF.RETRIEVED, B.TRUE)
        # reset analogs
        memory.nodes[token_mask, TF.ANALOG] = original_analogs
        # set analogs in recipient
        new_indicies = self.network.analog.get_analog_indices(new_analog)
        rec_analogs = original_analogs - torch.max(original_analogs).item()
        self.network.recipient().nodes[new_indicies, TF.ANALOG] += rec_analogs