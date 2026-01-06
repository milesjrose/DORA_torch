from .token_tensor import Token_Tensor
from ....enums import *
from ...network_params import Params
import torch
from logging import getLogger
logger = getLogger(__name__)

class UpdateOps:
    """ Class to perform update operations on the tensor."""
    def __init__(self, tokens: Token_Tensor, params: Params):
        """
        Initialize the UpdateOps object.
        Args:
            tokens: Token_Tensor - The tokens object.
        """
        self.tokens = tokens
        self.tensor = tokens.tensor
        self.cache = tokens.cache
        self.params = params

    def set_float(self, indices: torch.Tensor, features: list[TF], value: float = 0.0):
        """
        Initialise the float of the tokens.
        Args:
            indices: torch.Tensor - The indices of the tokens to set the float of.
            features: list[TF] - The features to initialise.
            value: float - The value to initialize the features to.
        """
        logger.info(f"Setting float of {len(indices)} tokens for features: {features} to {value}")
        logger.debug(f"Indices: {indices}")
        self.tensor[indices, features] = value
    
    def init_input(self, indices: torch.Tensor, refresh: float):
        """
        Initialise the input of the tokens.
        Args:
            indices: torch.Tensor - The indices of the tokens to initialise.
            refresh: float - The value to initialise the td_input to.
        """
        self.set_float(indices, [TF.TD_INPUT], refresh)
        features = [TF.BU_INPUT,TF.LATERAL_INPUT,TF.MAP_INPUT,TF.NET_INPUT]
        self.set_float(indices, features)
    
    def init_act(self, indices: torch.Tensor):
        """
        Initialise the act of the tokens.
        Args:
            indices: torch.Tensor - The indices of the tokens to initialise.
        """
        self.set_float(indices, [TF.ACT])
        self.init_input(indices, 0.0)
    
    def init_state(self, indices: torch.Tensor):
        """
        Initialise the state of the tokens.
        Args:
            indices: torch.Tensor - The indices of the tokens to initialise.
        """
        self.tensor[indices, TF.RETRIEVED] = False
        self.init_act(indices)
    
    def update_act(self, indices: torch.Tensor):
        """
        Update act of tokens in the given indices. 
        Based on params.gamma, params.delta, and params.HebbBias.
        Args:
            indices: torch.Tensor - The indices of the tokens to update.
        """
        if not torch.any(indices): return;
        net_input_types = [
            TF.TD_INPUT,
            TF.BU_INPUT,
            TF.LATERAL_INPUT
        ]
        gamma = self.params.gamma
        delta = self.params.delta
        HebbBias = self.params.HebbBias
        net_input = self.tensor[indices, net_input_types].sum(dim=1, keepdim=True)  # sum non mapping inputs
        net_input += (self.tensor[indices, TF.MAP_INPUT] * HebbBias).unsqueeze(1)   # Add biased mapping input, reshape to match
        acts = self.tensor[indices, TF.ACT]                                         # Get node acts
        delta_act = gamma * net_input.squeeze(1) * (1.1 - acts) - (delta * acts)    # Find change in act for each node
        acts += delta_act                                                           # Update acts
        
        self.tensor[(self.tensor[indices, TF.ACT] > 1.0), TF.ACT] = 1.0             # Limit activation to 1.0 or below
        self.tensor[(self.tensor[indices, TF.ACT] < 0.0), TF.ACT] = 0.0             # Limit activation to 0.0 or above                                      # update act

    def zero_lateral_input(self, indices: torch.Tensor):
        """
        Zero the lateral input of the tokens in the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to zero the lateral input of.
        """
        self.tensor[indices, TF.LATERAL_INPUT] = 0.0
    
    def update_inhibitor_input(self, indices: torch.Tensor):
        """
        Update the inhibitor input of the tokens in the given indices.
        """
        self.tensor[indices, TF.INHIBITOR_INPUT] += self.tensor[indices, TF.ACT]
    
    def reset_inhibitor(self, indices: torch.Tensor):
        """
        Reset the inhibitor input and act of the tokens in the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to reset the inhibitor input and act of.
        """
        self.tensor[indices, TF.INHIBITOR_INPUT] = 0.0
        self.tensor[indices, TF.INHIBITOR_ACT] = 0.0
    
    def update_inhibitor_act(self, indices: torch.Tensor):
        """
        Update the inhibitor act of the tokens in the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to update the inhibitor act of.
        """
        input = self.tensor[indices, TF.INHIBITOR_INPUT]
        threshold = self.tensor[indices, TF.INHIBITOR_THRESHOLD]
        nodes_to_update = (input >= threshold)                      # if inhib_input >= inhib_threshold
        # turn into global mask
        update_mask = torch.zeros_like(indices, dtype=torch.bool)
        update_mask[indices] = nodes_to_update
        # update
        self.tensor[update_mask, TF.INHIBITOR_ACT] = 1.0            # then set to 1
    
    def reset_maker_made_units(self, indices: torch.Tensor):
        """
        Reset the maker and made units of the tokens in the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to reset the maker and made units of.
        """
        self.tensor[indices, TF.MAKER_UNIT] = null
        self.tensor[indices, TF.MAKER_SET] = null
        self.tensor[indices, TF.MADE_UNIT] = null
        self.tensor[indices, TF.MADE_SET] = null