import torch
from .cache import Cache
from .token_tensor import Token_Tensor
from ....enums import *
from logging import getLogger
logger = getLogger(__name__)

class Analog_ops:
    """ Class to perform operations on analogs."""
    def __init__(self, tokens: Token_Tensor):
        """
        Initialize the Analog_ops object.
        Args:
            tokens: Token_Tensor - The tokens object.
        """
        self.tokens = tokens
        self.tensor = tokens.tensor
        self.cache = tokens.cache

    def new_analog_id(self) -> int:
        """
        Get a new analog id.
        """
        return torch.max(self.tensor[:, TF.ANALOG]) + 1
    
    def get_analog_indices(self, analog_number: int) -> torch.Tensor:
        """
        Get the indices of the tokens in the given analog.
        """
        return torch.where(self.tensor[:, TF.ANALOG] == analog_number)[0]
    
    def get_analogs_where(self, feature: TF, value) -> torch.Tensor:
        """
        Get any analogs that contain a token with a given feature and value.

        Args:
            feature (TF): The feature to check for.
            value (float): The value to check for.

        Returns:
            torch.Tensor: The analogs that contain a token with the given feature and value.
        """

        all_nodes_mask = self.cache.get_all_nodes_mask()                # Only non-deleted tokens
        matching_tokens = (self.tensor[all_nodes_mask, feature] == value)
        if not torch.any(matching_tokens):
            return torch.tensor([], dtype=torch.long)  # Return empty tensor (No matching tokens)
        
        matching_analog_ids = self.tensor[all_nodes_mask][matching_tokens, TF.ANALOG]   # Get the analog IDs of the matching tokens
        unique_analog_ids = torch.unique(matching_analog_ids)
        return unique_analog_ids
   
    def get_analogs_where_not(self, feature: TF, value) -> torch.Tensor:
        """
        Get any analogs that do not contain a token with a given feature and value.

        Args:
            feature (TF): The feature to check for.
            value (float): The value to check for.

        Returns:
            torch.Tensor: The analogs that do not contain a token with the given feature and value.
        """
        all_nodes_mask = self.cache.get_all_nodes_mask()               # Only non-deleted tokens
        non_matching_tokens = (self.tensor[all_nodes_mask, feature] != value)
        non_matching_analog_ids = self.tensor[all_nodes_mask][non_matching_tokens, TF.ANALOG] 
        unique_analog_ids = torch.unique(non_matching_analog_ids)       

        return unique_analog_ids
    
    def get_analogs_active(self) -> torch.Tensor:
        """
        Get all analogs that have at least one active token.
        """
        all_nodes_mask = self.cache.get_all_nodes_mask()
        active_tokens = (self.tensor[all_nodes_mask, TF.ACT] > 0.0)
        active_analog_ids = self.tensor[all_nodes_mask][active_tokens, TF.ANALOG]
        unique_analog_ids = torch.unique(active_analog_ids)

        return unique_analog_ids
    
    def move_analog(self, analog_number: int, to_set: Set):
        """
        Move the analog with the given number to the given set.
        Args:
            analog_number: int - The number of the analog to move.
            to_set: Set - The set to move the analog to.
        """
        logger.info(f"Moving analog {analog_number} to set {to_set}")
        indices = self.get_analog_indices(analog_number)
        self.tokens.move_tokens(indices, to_set)
        self.cache.cache_analogs()
    
    def copy_analog(self, analog_number: int, to_set: Set) -> int:
        """
        Copy the analog with the given number to the given set.
        Args:
            analog_number: int - The number of the analog to copy.
            to_set: Set - The set to copy the analog to.
        Returns:
            int - The number of the new analog.
        """
        logger.info(f"Copying analog {analog_number} to set {to_set}")
        indices = self.get_analog_indices(analog_number)
        # get a new analog number
        new_analog_number = self.new_analog_id()
        # Clone tokens and update both SET and ANALOG fields
        self.tokens.copy_tokens(indices, to_set)
        # Get names for the tokens
        self.tensor[indices, TF.ANALOG] = new_analog_number
        self.cache.cache_analogs()
        return new_analog_number