from functools import reduce
import torch
from ....enums import *
from logging import getLogger

logger = getLogger(__name__)

class TensorOperations:
    """
    Tensor operations for the Base_Set class.
    """
    def __init__(self, base_set):
        self.base_set = base_set

    def get_mask(self, token_type: Type) -> torch.Tensor:
        """
        Get the mask for a given token type.
        """
        mask = (self.base_set.lcl[:, TF.TYPE] == token_type)
        return mask
    
    def get_combined_mask(self, token_types: list[Type]) -> torch.Tensor:
        """
        Return combined mask of given types
        """
        if len(token_types) == 0:
            # Return all False mask if no types provided
            return torch.zeros(len(self.base_set.lcl), dtype=torch.bool)
        masks = [self.get_mask(token_type) for token_type in token_types]
        return reduce(torch.logical_or, masks)
    
    def get_count(self, token_type: Type=None, mask: torch.Tensor = None) -> int:
        """
        Get the count of tokens of a given type in the set.
        """
        if mask is None:
            mask = torch.ones(len(self.base_set.lcl), dtype=torch.bool)
        if token_type is None:
            return mask.sum().item()
        else:
            type_mask = self.get_mask(token_type)
            return (mask & type_mask).sum().item()
    
    def print(self, f_types=None):
        """
        Print the set.
        """
        logger.info(f"Printing set not implemmented yet :/")
        pass

    def print_tokens(self, f_types=None):
        """
        Print the tokens in the set.
        """
        logger.info(f"Printing tokens not implemmented yet :/")
        pass