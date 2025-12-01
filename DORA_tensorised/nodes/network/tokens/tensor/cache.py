import torch
from ....enums import *
from enum import IntEnum
from logging import getLogger
logger = getLogger(__name__)

class Analog_Cache(IntEnum):
    """ Enum for the analog cache."""
    ANALOG_NUMBERS = 0
    """ The analog numbers in the tensor."""
    ANALOG_SETS = 1
    """ The sets for each analog."""
    ANALOG_COUNTS = 2
    """ The counts for each analog."""
    ANALOG_ACTIVATIONS = 3
    """ The activations for each analog."""


class Cache:
    """ Class to cache tensor information."""
    def __init__(self, tensor: torch.Tensor):
        """
        Initialize the Set_Cache object.
        Args:
            tensor: torch.Tensor - The tensor of tokens.
        """
        self.tensor = tensor
        self.masks = {}
        self.analogs = torch.tensor([], dtype = tensor_type)
        """holds a tensor, with columns: [analog_number, analog_set, count, activation]"""
    
    def get_set_mask(self, set: Set) -> torch.Tensor:
        """
        Get the mask for the given set.
        Args:
            set: Set - The set to get the mask for.
        Returns:
            torch.Tensor - The mask for the given set.
        """
        if set not in self.masks:
            self.masks[set] = (self.tensor[:, TF.SET] == set)
        return self.masks[set]
    
    def get_set_indices(self, set: Set) -> torch.Tensor:
        """
        Get the indices for the given set.
        Args:
            set: Set - The set to get the indices for. If None, returns all indices.
        Returns:
            torch.Tensor - The indices for the given set.
        """
        if set is None:
            # Return all indices (all non-deleted tokens)
            return torch.where(self.get_all_nodes_mask())[0]
        indices = torch.where(self.get_set_mask(set))[0]
        return indices
    
    def get_all_nodes_mask(self) -> torch.Tensor:
        """
        Get the mask for all nodes in the tensor.
        Returns:
            torch.Tensor - The mask for all nodes in the tensor.
        """
        return (self.tensor[:, TF.DELETED] == B.FALSE)
    
    def get_set_count(self, set: Set) -> int:
        """
        Get the count of the given set.
        Args:
            set: Set - The set to get the count for.
        Returns:
            int - The count of the given set.
        """
        return self.get_set_mask(set).sum()
    
    def cache_sets(self, sets: list[Set]):
        """
        Update the cache for the given sets.
        Args:
            sets: list[Set] - The sets to update the cache for.
        """
        logger.info(f"Caching sets: {sets}")
        for set in sets:
            if set in self.masks:
                del self.masks[set]
            self.get_set_mask(set)
    
    def cache_analogs(self):
        """
        Cache the analogs in the tensor
        """
        logger.info(f"Caching analogs")
        # Get unique analog numbers and their counts
        analog_numbers, analog_counts = torch.unique(self.tensor[:, TF.ANALOG], return_counts=True)

        # Create a mapping from analog number to index
        analog_to_idx = {analog.item(): idx for idx, analog in enumerate(analog_numbers)}
        
        # Create index tensor for scatter
        analog_indices = torch.zeros(len(self.tensor), dtype=torch.long)
        for i, analog_num in enumerate(self.tensor[:, TF.ANALOG]):
            analog_indices[i] = analog_to_idx[analog_num.item()]
        
        # Use scatter_add_ to sum activations for each analog
        valid_mask = self.get_all_nodes_mask()
        analog_activations = torch.zeros(len(analog_numbers))
        analog_activations.scatter_add_(
            0, 
            analog_indices[valid_mask], 
            self.tensor[valid_mask, TF.ACT]
        )

        # get set for each analog (find first token with each analog number)
        analog_sets = torch.zeros(len(analog_numbers), dtype=torch.long)
        for i, analog_num in enumerate(analog_numbers):
            # Find first token with this analog number
            first_idx = torch.where(self.tensor[:, TF.ANALOG] == analog_num)[0][0]
            analog_sets[i] = self.tensor[first_idx, TF.SET].item()
        
        # combine into single tensor
        self.analogs = torch.column_stack((analog_numbers, analog_sets, analog_counts, analog_activations))